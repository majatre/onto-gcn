#!/usr/bin/env python
# coding: utf-8
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import copy
import time
import logging
import pickle
import numpy as np
import matplotlib, matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import torch
import itertools
from torch.autograd import Variable
import sklearn, sklearn.model_selection, sklearn.metrics
import numpy as np
from scipy import sparse
from models.mlp import MLP
from models.gnn import GCN
from models.utils import *
from data import datasets
from data.gene_graphs import GeneManiaGraph, OntologyGraph, StringDBGraph
from data.utils import record_result, randmap

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-task', type=str, default="PAM50")
args = parser.parse_args()
print(args)

path = "data/MBdata_original.csv"
df = pd.read_csv(path)

if args.task == 'DR':
    df = df[df.DR != '?']
    target = df.pop('DR')
elif args.task == 'ER':
    df = df[df.ER_Status != '?']
    target = df.pop('ER_Status')
    labels = {
        'pos': 0,
        'neg': 1
    }
    target = target.apply(lambda x: labels[x])
elif args.task == 'iC10':
    df = df[df.iC10 != '?']
    target = df.pop('iC10')
    labels = {
        '4ER-': 4,
        '4ER+': 0
    }
    target = target.apply(lambda x: labels[x] if x in labels else int(x))
elif args.task == 'PAM50':
    df = df[df.Pam50Subtype != '?']
    target = df.pop('Pam50Subtype')
    pam50_lables = {
        'Normal': 0,
        'LumA': 1,
        'LumB': 2,
        'Basal': 3,
        'Her2': 4
    }
    target = target.apply(lambda x: pam50_lables[x])
else:
    print("Task not specified.")
    

features = df.filter(regex='^GE.*')
features = features.astype('float64')
print(features.shape)

def normalize(df):
    r = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        if max_value != min_value:
          r[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return r

features = df.filter(regex='^GE.*')
features = normalize(features)
features.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
features = features.rename(columns={g: g[3:] for g in list(features.columns)})


                                                                           

# Setup the results dictionary
filename = "experiments/results/res_gat_A_" + args.task + ".pkl"
try:
    results = pickle.load(open(filename, "rb"), encoding='latin1')
    print("Loaded Checkpointed Results")
except Exception as e:
    results = pd.DataFrame(columns=['auc', 'acc', 'f1', 'gene', 'model', 'graph',  'emb', 'num_genes', 'seed', 'train_size', 'time_elapsed'])
    print("Created a New Results Dictionary")

#gene_graph = GeneManiaGraph()OntologyGraph(neighbors=500, embeddings_name='dl2vec'), OntologyGraph(neighbors=100, embeddings_name='dl2vec'),
# gene_graphs = [OntologyGraph(neighbors=100, embeddings_name='dl2vec', randomize=False), GeneManiaGraph(), StringDBGraph(datastore="./data")]StringDBGraph(datastore="./data"),  
gene_graphs = [OntologyGraph(neighbors=10, embeddings_name='dl2vec', randomize=False, gene_names=list(features.columns), relabel_genes=False),
    # OntologyGraph(neighbors=30, embeddings_name='dl2vec', randomize=True, gene_names=list(features.columns), relabel_genes=False),
    # OntologyGraph(neighbors=30, embeddings_name='el', randomize=False, gene_names=list(features.columns), relabel_genes=False),
    # GeneManiaGraph(), StringDBGraph(datastore="./data")
    ]
# gene_graphs = [
#         OntologyGraph(neighbors=n, embeddings_name=emb) 
#         for n in [500, 100, 30, 10] 
#         for emb in ['el', 'dl2vec', 'opa2vec', 'opa2vec_go']]

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


    
for gene_graph in gene_graphs:

    search_num_genes=['all'] #, 100, 1000, 2000] #[4000, 8000] #, 8000, 16300]1000,
    test_size=400
    search_train_size=[100, 1500] #, 50]100,
    cuda = torch.cuda.is_available()
    trials=3
    search_genes = ["ESR1"]

    # Create the set of all experiment ids and see which are left to do 'GCN7','GAT', 'GAT',
    model_names = [ 'GAT3']#, 'GCN'] #'GCN3','GCN4', 'GCN5', 'MLP64_lr4'] #[model.name for model in models]'MLP64','MLP64_lr4' ] #
    graph_names = [gene_graph.graph_name]
    columns = ["gene", "model", "graph", "num_genes", "train_size", "seed"]
    all_exp_ids = [x for x in itertools.product(search_genes, model_names, graph_names, search_num_genes, search_train_size, range(trials))]
    all_exp_ids = pd.DataFrame(all_exp_ids, columns=columns)
    all_exp_ids.index = ["-".join(map(str, tup[1:])) for tup in all_exp_ids.itertuples(name=None)]
    results_exp_ids = results[columns].copy()
    results_exp_ids.index = ["-".join(map(str, tup[1:])) for tup in results_exp_ids.itertuples(name=None)]
    intersection_ids = all_exp_ids.index.intersection(results_exp_ids.index)
    todo = all_exp_ids.drop(intersection_ids).to_dict(orient="records")
    print("todo: " + str(len(todo)))
    print("done: " + str(len(results)))



    for row in todo:
        import gc
        gc.collect()
        # print(row)
        start_time = time.time()
        gene = row["gene"]
        model_name = row["model"]
        graph_name = row["graph"]
        seed = row["seed"]
        num_genes = row["num_genes"] #if row["num_genes"] < 10000 else 16300
        train_size = row["train_size"]
        node_embeddings = 'dl2vec'
        # model = [copy.deepcopy(model) for model in models if model.name == row["model"]][0]
        experiment = {
            "gene": gene,
            "model": model_name,
            "graph": graph_name,
            "emb": node_embeddings,
            "num_genes": num_genes,
            "train_size": train_size,
            "seed": seed,
        }
        print(experiment)

        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(features, 
                                                                                target.to_numpy(), 
                                                                                stratify=target.to_numpy(),
                                                                                train_size=train_size,
                                                                                test_size=test_size,
                                                                                shuffle=True,
                                                                                random_state=seed
                                                                                )
        if gene_graph.randomize:
            print("Randomizing the graph")
            gene_graph.nx_graph = nx.relabel.relabel_nodes(gene_graph.nx_graph, randmap(gene_graph.nx_graph.nodes))

        if num_genes == 'all':
            if 'MLP' in model_name:
                x_train = X_train.copy()
                x_test = X_test.copy()
                adj = None
            else:
                neighbors = gene_graph.nx_graph
                intersection_nodes = np.intersect1d(X_train.columns, neighbors.nodes)
                x_train = X_train[list(intersection_nodes)].copy()
                x_test = X_test[list(intersection_nodes)].copy()

                toremove = set(neighbors.nodes)
                toremove = toremove.difference(intersection_nodes)
                neighbors.remove_nodes_from(toremove)

                adj = sparse.csr_matrix(nx.to_numpy_matrix(neighbors))

        else:
            gene = 'ESR1'
            neighbors = gene_graph.bfs_sample_neighbors(gene, num_genes*1.5)
            intersection_nodes = np.intersect1d(X_train.columns, neighbors.nodes)

            x_train = X_train[list(intersection_nodes)[:num_genes]].copy()
            x_test = X_test[list(intersection_nodes)[:num_genes]].copy()

            toremove = set(neighbors.nodes)
            toremove = toremove.difference(intersection_nodes)
            neighbors.remove_nodes_from(toremove)

            adj = sparse.csr_matrix(nx.to_numpy_matrix(neighbors))
            print(adj.shape)
        

        
        if node_embeddings == 'random':
            emb_vectors = []
            for gene in x_train.columns:
                emb_vectors.append(list(np.random.random(16)))
        else:
            emb_path = "./data/datastore/" + node_embeddings + "_vectors.csv"
            embedding_data = pd.read_csv(emb_path)
            embedding_data = embedding_data.set_index('Gene_ID')
            emb_vectors = []
            missing = 0
            for gene in x_train.columns:
                if gene in embedding_data.index:
                    emb_vectors.append(list(embedding_data.loc[gene]))
                else:
                    emb_vectors.append(list(np.random.random(embedding_data.shape[1])))
                    missing += 1
            print("Missing embeddings ", missing)
        emb_vectors = pd.DataFrame(emb_vectors)
        if torch.cuda.is_available():
            emb_vectors = torch.FloatTensor(np.array(emb_vectors)).cuda()
        else:
            emb_vectors = torch.FloatTensor(np.array(emb_vectors))


        if model_name == 'GCN':
            model = GCN(name="GCN_noemb_lay1_pre2_chan64_dropout_agg_hierarchical", #_lay3_chan64_emb32_dropout_agg_hierarchy", 
                                dropout=True, 
                                cuda=torch.cuda.is_available(),
                                gnn="GCN",
                                num_layer=1,
                                prepool_extralayers=2,
                                channels=64, 
                                embedding=64, # sdict["embedding"], 
                                aggregation="hierarchical",
                                lr=0.001,
                                num_epochs=100,
                                patience=30,
                                verbose=True,
                                seed=seed,
                                train_valid_split=0.8
                                )
        elif model_name == 'GAT':
            model = GCN(name="GAT_noemb_lay1_pre0_heads2_chan16", #_lay3_chan64_emb32_dropout_agg_hierarchy", 
                                dropout=False, 
                                cuda=torch.cuda.is_available(),
                                gnn="GAT",
                                num_layer=1,
                                prepool_extralayers=0,
                                gat_heads=2,
                                channels=16, 
                                embedding=16, # sdict["embedding"], 
                                aggregation=None,
                                lr=0.001,
                                num_epochs=100,
                                patience=30,
                                batch_size=10,
                                verbose=True,
                                seed=seed,
                                train_valid_split=0.8
                                )
        elif model_name == 'GAT2':
            model = GCN(name="GAT_noemb_lay1_pre0_chan32", #_lay3_chan64_emb32_dropout_agg_hierarchy", 
                                dropout=False, 
                                cuda=torch.cuda.is_available(),
                                gnn="GAT",
                                num_layer=1,
                                prepool_extralayers=0,
                                channels=32, 
                                embedding=32, # sdict["embedding"], 
                                aggregation=None,
                                lr=0.001,
                                num_epochs=100,
                                patience=30,
                                verbose=True,
                                seed=seed,
                                train_valid_split=0.8
                                )
        elif model_name == 'GAT3':
            model = GCN(name="GAT_noemb_lay1_pre0_chan64", #_lay3_chan64_emb32_dropout_agg_hierarchy", 
                                dropout=False, 
                                cuda=torch.cuda.is_available(),
                                gnn="GAT",
                                num_layer=1,
                                prepool_extralayers=0,
                                channels=64, 
                                embedding=64, # sdict["embedding"], 
                                aggregation=None,
                                lr=0.001,
                                num_epochs=100,
                                patience=30,
                                verbose=True,
                                seed=seed,
                                train_valid_split=0.8
                                )

        try:
            # print(x_train.shape,  y_train.shape, adj.shape)

            model.fit(x_train, y_train, adj=adj) #, ontology_vectors=emb_vectors)

            with torch.no_grad():
                model.eval()
                torch.cuda.empty_cache()
                
                y_hat = model.predict(x_test)
                y_hat = np.argmax(y_hat, axis=1)
                # auc = sklearn.metrics.roc_auc_score(y_test, np.asarray(y_hat).flatten(), multi_class='ovo')
                acc = sklearn.metrics.accuracy_score(y_test, np.asarray(y_hat).flatten())
                f1 = sklearn.metrics.f1_score(y_test, np.asarray(y_hat).flatten(), average='macro')

                experiment["model"] = model.name
                experiment["auc"] = 0
                experiment["acc"] = acc
                experiment["f1"] = f1
                experiment["num_genes"] = len(x_train.columns)

                experiment["time_elapsed"] = str(time.time() - start_time)
                results = record_result(results, experiment, filename)
                print(experiment)
        except Exception as e: 
            print("------------------------------------")
            print(e)
            logging.error(logging.traceback.format_exc())
            print("------------------------------------")

        # cleanup
        model.best_model = None  
        del model
        torch.cuda.empty_cache()