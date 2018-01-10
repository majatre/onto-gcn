import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
import graphLayer
from torchvision import transforms, utils
import os

class EmbeddingLayer(nn.Module):

    def __init__(self, nb_emb, emb_size=32):

        self.emb_size = emb_size
        super(EmbeddingLayer, self).__init__()

        # The embeddings
        self.emb_size = emb_size
        self.emb = nn.Parameter(torch.rand(nb_emb, emb_size))

    def forward(self, x):
        emb = x * self.emb
        return emb



class AttentionLayer(nn.Module):

    def __init__(self, in_dim):

        self.in_dim = in_dim
        super(AttentionLayer, self).__init__()

        # The view vector.
        self.attn = nn.Linear(self.in_dim, 1)
        self.temperature = 1.

    def forward(self, x):
        nb_examples, nb_channels, nb_nodes = x.size()
        x = x.permute(0, 2, 1).contiguous()  # from ex, ch, node -> ex, node, ch
        x = x.view(-1, nb_channels)

        # attn_weights = F.softmax(self.attn(x), dim=1)# Should be able to do that,
        # I have some problem with pytorch right now, so I'm doing i manually. Also between you and me, the pytorch example for attention sucks.
        attn_weights = torch.exp(self.attn(x)*self.temperature)
        attn_weights = attn_weights.view(nb_examples, nb_nodes, 1)
        attn_weights = attn_weights / attn_weights.sum(dim=1).unsqueeze(-1) # normalizing

        x = x.view(nb_examples, nb_nodes, nb_channels)
        attn_applied = x * attn_weights
        attn_applied = attn_applied.sum(dim=1)
        #print attn_weights[0].max()

        return attn_applied


# Create a module for the CGN:
class CGN(nn.Module):

    def __init__(self, nb_nodes, input_dim, channels, adj, out_dim,
                 on_cuda=True, add_residual=False, attention_layer=0, add_emb=None, transform_adj=None):
        super(CGN, self).__init__()

        if transform_adj is None:
            transform_adj = []

        self.my_layers = []
        self.out_dim = out_dim
        self.on_cuda = on_cuda
        self.add_residual = add_residual
        self.nb_nodes = nb_nodes
        self.nb_channels = channels
        self.attention_layer = attention_layer
        self.add_emb = add_emb

        if add_emb:
            print "Adding node embeddings."
            self.emb = EmbeddingLayer(nb_nodes, add_emb)
            input_dim = self.emb.emb_size

        dims = [input_dim] + channels

        print "Constructing the network..."
        # The normal layer
        layers = []
        for c_in, c_out in zip(dims[:-1], dims[1:]):
            layer = nn.Conv1d(c_in, c_out, 1, bias=True)
            layers.append(layer)
        self.my_layers = nn.ModuleList(layers)

        # The convolutional layer
        convs = []

        for i in range(len(channels)):
            # transformation to apply at each layer.
            transform_tmp = transforms.Compose([foo(please_ignore=i == 0, unique_id=i) for foo in transform_adj])
            convs.append(graphLayer.CGNLayer(nb_nodes, adj, on_cuda, transform_tmp))
            adj = convs[-1].adj

        self.my_convs = nn.ModuleList(convs)

        # The logistic layer
        logistic_layer = []
        if not channels: # Only have one layer
            logistic_in_dim = [nb_nodes * input_dim]
        elif not add_residual: # Adding a final logistic regression.
            if attention_layer > 0:
                logistic_in_dim = [channels[-1] * attention_layer]  # Changed
            else:
                logistic_in_dim = [nb_nodes * channels[-1]] # Changed here
        else:
            print "Adding skip connections..."
            if attention_layer > 0:
                logistic_in_dim = [d * nb_nodes for d in dims]
            else:
                logistic_in_dim = [d * attention_layer for d in dims]

        for d in logistic_in_dim:
            layer = nn.Linear(d, out_dim)
            logistic_layer.append(layer)

        self.my_logistic_layers = nn.ModuleList(logistic_layer)
        print "Done!"

        if attention_layer > 0:
            print "Adding {} attentions layer.".format(attention_layer)
            self.att = nn.ModuleList([AttentionLayer(channels[-1])] * attention_layer)

    def forward(self, x):

        out = None
        nb_examples, nb_nodes, nb_channels = x.size()
        if self.add_emb:
            x = self.emb(x)

        x = x.permute(0, 2, 1).contiguous()# from ex, node, ch, -> ex, ch, node

        # Do graph convolution for all
        for num, [conv, layer] in enumerate(zip(self.my_convs, self.my_layers)):

            if self.add_residual: # skip connection
                if out is None:
                    out = self.my_logistic_layers[num](x.view(nb_examples, -1))
                else:
                    out += self.my_logistic_layers[num](x.view(nb_examples, -1))

            x = conv(x) # conv
            x = F.relu(layer(x))  # or relu, sigmoid...

        # agregate the attention on the last layer.
        if self.attention_layer > 0:
            x = torch.stack([att(x) for att in self.att], dim=-1)

        if out is None:
            out = self.my_logistic_layers[-1](x.view(nb_examples, -1))
        else:
            out += self.my_logistic_layers[-1](x.view(nb_examples, -1))

        return out

# Create a module for MLP
class MLP(nn.Module):
    def __init__(self,input_dim, channels, out_dim=None, on_cuda=True):
        super(MLP, self).__init__()

        self.my_layers = []
        self.out_dim = out_dim
        self.on_cuda = on_cuda

        dims = [input_dim] + channels

        print "Constructing the network..."
        layers = []
        for c_in, c_out in zip(dims[:-1], dims[1:]):
            layer = nn.Linear(c_in, c_out)
            layers.append(layer)
        self.my_layers = nn.ModuleList(layers)

        if channels:
            self.last_layer = nn.Linear(channels[-1], out_dim)
        else:
            self.last_layer = nn.Linear(input_dim, out_dim)

        print "Done!"

    def forward(self, x):
        nb_examples, nb_nodes, nb_channels = x.size()

        x = x.permute(0, 2, 1).contiguous()  # from ex, node, ch, -> ex, ch, node
        for layer in self.my_layers:
            x = F.relu(layer(x.view(nb_examples, -1)))  # or relu, sigmoid...

        x = self.last_layer(x.view(nb_examples, -1))

        return x

# TODO: have a LCGLayer class to simplify stuff.
class LCG(nn.Module):
    def __init__(self,input_dim, A, channels=16, out_dim=2, on_cuda=False, num_layers = 1, arg_max = -1):
        super(LCG, self).__init__()

        print "Bip bop I'm Francis and I'm lazy, I need to use all the adjs."
        A = A[0] # just use first graph # TODO: add all of them.
        
        self.my_layers = []
        self.out_dim = out_dim
        self.on_cuda = on_cuda
        self.nb_nodes = A.shape[0]
        self.num_layers = num_layers

        self.nb_channels = [channels] # we only support 1 layer for now.

        print "Constructing the network..."   
        self.max_edges = sorted((A > 0.).sum(0))[arg_max]

        print "Each node will have {} edges.".format(self.max_edges)


        # Get the list of all the edges. All the first index is 0, we fix that later
        edges_np = [np.asarray(np.where(A[i:i+1] > 0.)).T for i in range(len(A))]


        # pad the edges, so they all nodes have the same number of edges. help to automate everything.
        edges_np = [np.concatenate([x, [[0, self.nb_nodes]] * (self.max_edges - len(x))]) if len(x) < self.max_edges
                    else x[:self.max_edges] if len(x) > self.max_edges # Some Nodes have to many connection!
                    else x
                    for i, x in enumerate(edges_np)]

        # fix the index that was all 0.
        for i in range(len(edges_np)):
            edges_np[i][:, 0] = i


        edges_np = np.array(edges_np).reshape(-1, 2)
        edges_np = edges_np[:, 1:2]

        self.edges = torch.LongTensor(edges_np)
        self.super_edges = torch.cat([self.edges] * channels)

        # we add a weight the fake node (that we introduced in the padding)
        my_weights = [nn.Parameter(torch.rand(self.edges.shape[0], channels), requires_grad=True) for _ in range(num_layers)]
        self.my_weights = nn.ParameterList(my_weights)

        last_layer = nn.Linear(input_dim * channels, out_dim)
        self.my_logistic_layers = nn.ModuleList([last_layer])

        #self.register_buffer('edges', self.edges)

        print "Done!"

    def GraphConv(self, x, edges, batch_size, weights):
        
        edges = edges.contiguous().view(-1)
        useless_node = Variable(torch.zeros(x.size(0), 1, x.size(2)))

        if self.on_cuda:
            edges = edges.cuda()
            weights = weights.cuda()
            useless_node = useless_node.cuda()

        x = torch.cat([x, useless_node], 1) # add a random filler node
        tocompute = torch.index_select(x, 1, Variable(edges)).view(batch_size, -1, weights.size(-1))

        conv = tocompute * weights
        conv = conv.view(-1, self.nb_nodes, self.max_edges, weights.size(-1)).sum(2)
        return F.relu(conv)

    def forward(self, x):

        nb_examples, nb_nodes, nb_channels = x.size()
        edges = Variable(self.super_edges, requires_grad=False)

        if self.on_cuda:
            edges = edges.cuda()

        for i in range(self.num_layers):
            x = self.GraphConv(x, edges.data, nb_examples, self.my_weights[i])

        x = self.my_logistic_layers[-1](x.view(nb_examples, -1))
        #x = F.softmax(x)

        return x



#spectral graph conv
class SGC(nn.Module):
    def __init__(self,input_dim, A, channels=1, out_dim=2, on_cuda=False, num_layers = 1, arg_max = -200):
        super(SGC, self).__init__()

        print "Bip bop I'm Francis and I'm lazy, I need to use all the adjs."
        A = A[0] # just use first graph
        
        self.my_layers = []
        self.out_dim = out_dim
        self.on_cuda = on_cuda
        self.nb_nodes = input_dim
        self.num_layers = num_layers

        self.channels = 1#channels
        #dims = [input_dim] + channels

        def if_cuda(x):
            return x.cuda() if self.on_cuda else x

        print "Constructing the eigenvectors..."   
        
        D = np.diag(A.sum(axis=1))
        self.L = D-A
        self.L = torch.FloatTensor(self.L)
        self.L = if_cuda(self.L)
        
        eg = load_eigenvectors("",self.L)
        if eg != None:
            self.g, self.V = if_cuda(eg[0]),if_cuda(eg[1])
        else:
            self.g, self.V = torch.eig(self.L, eigenvectors=True)
            save_eigenvectors("",self.L, self.g, self.V)
        
        self.V = if_cuda(self.V.cpu().half())
        self.g = if_cuda(self.g.cpu().half())
        
        print "self.nb_nodes", self.nb_nodes
        self.F = nn.Parameter(if_cuda(torch.rand(self.nb_nodes, self.nb_nodes).half()), requires_grad=True)
        self.my_bias = nn.Parameter(if_cuda(torch.zeros(self.nb_nodes, channels)), requires_grad=True)

        
        last_layer = nn.Linear(self.nb_nodes * self.channels, out_dim).half()
        self.my_logistic_layers = nn.ModuleList([last_layer])

        print "Done!"

    def forward(self, x):

        nb_examples, nb_nodes, nb_channels = x.size()

        def if_cuda(x):
            return x.cuda().half() if self.on_cuda else x.half()
        
        x = if_cuda(x.cpu())
        Vx = torch.matmul(torch.transpose(Variable(self.V), 0,1),x)
        FVx = torch.matmul(self.F, Vx)
        VFVx = torch.matmul(Variable(self.V),FVx)
        x = VFVx
        
        
        x = self.my_logistic_layers[-1](x.view(nb_examples, -1))
        x = F.softmax(x, dim=1)
        
        return x


def get_eigenvectors_filename(name,L):
    cachepath="./cache/"
    matrix_hash=str(hash(L.cpu().numpy().tostring()))
    return cachepath + matrix_hash + ".npz"

def load_eigenvectors(name,L):
    filename = get_eigenvectors_filename(name,L)
    if os.path.isfile(filename):
        print "loading", filename
        eg = np.load(open(filename))
        return (torch.FloatTensor(eg["g"]),torch.FloatTensor(eg["V"]))
    
def save_eigenvectors(name,L,g,V):
    filename = get_eigenvectors_filename(name,L)
    print "saving", filename
    return np.savez(open(filename,'w+'),g=g.cpu().numpy(),V=V.cpu().numpy())


def get_model(opt, dataset, nb_class):
    """
    Return a model based on the options.
    :param opt:
    :param dataset:
    :param nb_class:
    :return:
    """

    model = opt.model
    num_channel = opt.num_channel
    num_layer = opt.num_layer
    on_cuda = opt.cuda
    skip_connections = opt.skip_connections

    transform = graphLayer.get_transform(opt)

    if model == 'cgn':
        # To have a feel of the model, please take a look at cgn.ipynb
        my_model = CGN(dataset.nb_nodes, 1, [num_channel] * num_layer, dataset.get_adj(), nb_class,
                       on_cuda=on_cuda, add_residual=skip_connections, attention_layer=opt.attention_layer,
                       add_emb=opt.use_emb,  transform_adj=transform)

    elif model == 'mlp':
        my_model = MLP(dataset.nb_nodes, [num_channel] * num_layer, nb_class,
                       on_cuda=on_cuda)  # TODO: add a bunch of the options

    elif model == 'lcg':
        my_model = LCG(dataset.nb_nodes, dataset.get_adj(), out_dim=nb_class,
                       on_cuda=on_cuda, channels=num_channel, num_layers=num_layer)  # TODO: add a bunch of the options

    elif model == 'sgc':
        my_model = SGC(dataset.nb_nodes, dataset.get_adj(), out_dim=nb_class,
                       on_cuda=on_cuda, channels=num_channel, num_layers=num_layer)
    else:
        raise ValueError

    return my_model
    