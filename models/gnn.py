"""A GCN.py class extended to support also the GAT message passing."""

import logging
import time
import itertools
import sklearn
import sklearn.model_selection
import sklearn.metrics
import sklearn.linear_model
import sklearn.neural_network
import sklearn.tree
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from scipy import sparse
from models.utils import *
from models.models import Model
from models.gcn_layers import *
import scipy.sparse
from torch_geometric.utils.convert import from_scipy_sparse_matrix
from torch_geometric.nn.conv.gat_conv import GATConv
from torch_geometric.nn import TopKPooling
from torch_geometric.data import *

class GCN(Model):
    def __init__(self, **kwargs):
        super(GCN, self).__init__(**kwargs)

    def setup_layers(self):
        self.master_nodes = 0
        self.in_dim = 1
        self.out_dim = len(np.unique(self.y))

        if (self.adj is None):
            raise Exception("adj must be specified for GCN")
        self.adj = scipy.sparse.csr_matrix(self.adj)
        self.adjs, self.centroids = setup_aggregates(self.adj, self.num_layer, self.X, aggregation=self.aggregation, agg_reduce=self.agg_reduce, verbose=self.verbose)
        if len(self.adjs)==0:
            self.adjs = [self.adj]
        self.nb_nodes = self.X.shape[1]

        if (self.ontology_vectors is not None):
            print("Onto embedding layer")
            self.add_onto_embedding_layer()
            self.in_dim = self.emb.emb_size
        elif self.embedding:
            self.add_embedding_layer()
            self.in_dim = self.emb.emb_size
        self.dims = [self.in_dim] + self.channels
        self.add_graph_convolutional_layers()
        self.add_logistic_layer()
        self.add_gating_layers()
        self.add_dropout_layers()

        if self.attention_head:
            self.attention_layer = AttentionLayer(self.channels[-1], self.attention_head)

        torch.manual_seed(self.seed)
        if self.on_cuda:
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            self.cuda()

    def forward(self, x):
        nb_examples, nb_nodes, nb_channels = x.size()


        if self.embedding:
            x = self.emb(x)

            
        if self.gnn == "GAT":
            device = torch.device('cuda')
            edge_index, edge_weight = from_scipy_sparse_matrix(self.adjs[0])
            data = Batch.from_data_list([Data(x=g, edge_index=edge_index).to(device) for g in x]).to(device)
            x, edge_index, batch = data.x, data.edge_index, data.batch

        for i, [conv, gate, dropout] in enumerate(zip(self.conv_layers, self.gating_layers, self.dropout_layers)):

            for prepool_conv in self.prepool_conv_layers[i]:
                if self.gnn == "GCN":
                    x = prepool_conv(x)
                elif self.gnn == "GAT":
                    x = prepool_conv(x, edge_index)

            if self.gnn == "GCN":
                x = conv(x)
            elif self.gnn == "GAT":
                x = conv(x, edge_index)
                # print(x.shape)
                # x, edge_index, _, batch, _, _ = pool(x, edge_index, None, batch).to(device)F.relu()


            if dropout is not None:
                id_to_keep = dropout(torch.FloatTensor(np.ones((x.size(0), x.size(1))))).unsqueeze(2)
                if self.on_cuda:
                    id_to_keep = id_to_keep.cuda()
                x = x * id_to_keep

        # Do attention pooling here
        if self.attention_head:
            x = self.attention_layer(x)[0]

        x = self.my_logistic_layers[-1](x.view(nb_examples, -1))
        return x


    def add_embedding_layer(self):
        self.emb = EmbeddingLayer(self.nb_nodes, self.embedding)

    def add_onto_embedding_layer(self):
        if self.expression_scaling:
            self.emb = EmbeddingLayerScaling(self.nb_nodes, self.ontology_vectors, self.ontology_vectors.shape[1], self.embedding)
        else:
            self.emb = EmbeddingLayerFromOntology(self.nb_nodes, self.ontology_vectors, self.ontology_vectors.shape[1], self.embedding)

    def add_dropout_layers(self):
        self.dropout_layers = [None] * (len(self.dims) - 1)
        if self.dropout:
            self.dropout_layers = nn.ModuleList([torch.nn.Dropout(int(self.dropout)*min((id_layer+1) / 10., 0.4)) for id_layer in range(len(self.dims)-1)])

    def add_graph_convolutional_layers(self):
        convs = []
        prepool_convs = nn.ModuleList([])
        # top_k_pooling = nn.ModuleList([])
        # if len(self.dims) == 1:
        #     self.dims = self.dims + self.dims
        for i, [c_in, c_out] in enumerate(zip(self.dims[:-1], self.dims[1:])):
            # transformation to apply at each layer.
            extra_layers = []
            for _ in range(self.prepool_extralayers):
                if self.gnn == "GCN":
                    extra_layer = GCNLayer(self.adjs[i], c_in, c_in, self.on_cuda, i, torch.LongTensor(np.array(range(self.adjs[i].shape[0]))))
                elif self.gnn == "GAT":
                    extra_layer = GATConv(c_in, c_in, heads=self.gat_heads, concat=True, negative_slope=0.2, dropout=0, bias=True)
                extra_layers.append(extra_layer)

            prepool_convs.append(nn.ModuleList(extra_layers))

            if self.gnn == "GCN":
                layer = GCNLayer(self.adjs[i], c_in, c_out, self.on_cuda, i, torch.tensor(self.centroids[i]))
            elif self.gnn == "GAT":
                layer = GATConv(c_in, c_out, heads=self.gat_heads, concat=True, negative_slope=0.2, dropout=0, bias=True)
                # top_k_pooling.append(TopKPooling(c_out, ratio=0.5))

            convs.append(layer)
        self.conv_layers = nn.ModuleList(convs)
        self.prepool_conv_layers = prepool_convs
        # self.top_k_pooling = top_k_pooling

    def add_gating_layers(self):
        if self.gating > 0.:
            gating_layers = []
            for c_in in self.channels:
                gate = ElementwiseGateLayer(c_in)
                gating_layers.append(gate)
            self.gating_layers = nn.ModuleList(gating_layers)
        else:
            self.gating_layers = [None] * (len(self.dims) - 1)

    def add_logistic_layer(self):
        logistic_layers = []
        if self.attention_head > 0:
            logistic_in_dim = [self.attention_head * self.dims[-1]]
        elif self.gnn == "GAT":
            logistic_in_dim = [int(self.adjs[0].shape[0]) * self.dims[-1]]
        else:
            logistic_in_dim = [self.adjs[-1].shape[0] * self.dims[-1]]
        for d in logistic_in_dim:
            layer = nn.Linear(d, self.out_dim)
            logistic_layers.append(layer)
        self.my_logistic_layers = nn.ModuleList(logistic_layers)

    def get_representation(self):
        def add_rep(layer, name, rep):
            rep[name] = {'input': layer.in_features, 'output': layer.out_features}

        representation = {}

        # if self.embedding:
        #     add_rep(self.emb, 'emb', representation)

        for i, [layer, gate] in enumerate(zip(self.conv_layers, self.gating_layers)):

            if self.gating > 0.:
                add_rep(layer, 'layer_{}'.format(i), representation)
                add_rep(gate, 'gate_{}'.format(i), representation)

            else:
                add_rep(layer, 'layer_{}'.format(i), representation)

        add_rep(self.my_logistic_layers[-1], 'logistic', representation)

        if self.attention_head:
            representation['attention'] = {'input': self.attention_layer.input[0].cpu().data.numpy(),
                         'output': [self.attention_layer.output[0].cpu().data.numpy(), self.attention_layer.output[1].cpu().data.numpy()]}

        return representation

    # because of the sparse matrices.
    def load_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            try:
                own_state[name].copy_(param)
            except (AttributeError, RuntimeError):
                pass # because of the sparse matrices.
