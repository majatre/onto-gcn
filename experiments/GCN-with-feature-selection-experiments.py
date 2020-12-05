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
from models.gcn import GCN
from models.utils import *
from data import datasets
from data.gene_graphs import GeneManiaGraph, OntologyGraph, StringDBGraph
from data.utils import record_result, randmap
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-task', type=str, default="PAM50")
args = parser.parse_args()
print(args)

path = "data/MBdata_all.csv"
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

from sklearn.metrics.pairwise import cosine_distances
def load_gene_embeddings(name = "el", reference_gene = "ESR1"):
  path = "./data/datastore/" + name+"_vectors.csv"
  embedding_data = pd.read_csv(path)
  labels = embedding_data.pop('Gene_ID')
  print(embedding_data.shape)

  df = pd.read_csv(path)
  df = df.set_index('Gene_ID')

  reference_emb = df.loc[reference_gene].to_numpy().reshape(1, -1)
  cos_distances = {}

  for index, row in df.iterrows():
      similarity = cosine_distances(reference_emb, row.to_numpy().reshape(1, -1))
      cos_distances[index] = similarity

  cos_distances_sorted = {k: v for k, v in sorted(cos_distances.items(), key=lambda item: item[1])}
  return cos_distances_sorted


# Setup the results dictionary
filename = "experiments/results/fs_MB_A_" + args.task + ".pkl"
try:
    results = pickle.load(open(filename, "rb"), encoding='latin1')
    print("Loaded Checkpointed Results")
except Exception as e:
    results = pd.DataFrame(columns=['auc', 'acc', 'f1', 'gene', 'model', 'graph', 'num_genes', 'seed', 'train_size', 'time_elapsed'])
    print("Created a New Results Dictionary")

gene_graphs = [ OntologyGraph(neighbors=10, embeddings_name='el', randomize=False),
#                  OntologyGraph(neighbors=30, embeddings_name='el', randomize=False),
#                  OntologyGraph(neighbors=30, embeddings_name='dl2vec', randomize=False),
                 OntologyGraph(neighbors=10, embeddings_name='dl2vec', randomize=True)]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

gene = 'ESR1'
cos_distances_sorted = load_gene_embeddings(name='el', reference_gene=gene)
top_genes = [g for g,v in cos_distances_sorted.items() if g in list(features.columns)]
print(top_genes[:10])
    
for gene_graph in gene_graphs:

    search_num_genes=[ 10, 100, 200, 500, 1000, 2000, 4000, 8000, 16000, 20000] #'all'] #, 100, 1000, 2000] #[4000, 8000] #, 8000, 16300]1000,10, 100, 200, 500, 1000, 2000, 4000, 8000, 16000,
    test_size=400
    search_train_size=[100, 1500] #, 50]100,
    cuda = torch.cuda.is_available()
    trials=3
    search_genes = ["ESR1"]

    # Create the set of all experiment ids and see which are left to do 'MLP1','MLP2', 'MLP2.1', 'MLP3', 'MLP4', , 'GCN', 'GCN16', 'GCN2','GCN3.0', 'GCN3.1', 'GCN5',  'GCN4'
    model_names = [ 'GCN5',  'GCN', 'GCN16', 'GCN2', 'MLP1','MLP2', 'MLP2.1',] #,  'MLP3', 'MLP4''MLP64_lr4'] #'GCN3','GCN4', 'GCN5', 'MLP64_lr4'] #[model.name for model in models]
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
        # model = [copy.deepcopy(model) for model in models if model.name == row["model"]][0]
        experiment = {
            "gene": gene,
            "model": model_name,
            "graph": graph_name,
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
            columns = list(X_train.columns)
            random.shuffle(columns)
            x_train = X_train[columns[:num_genes]].copy()
            x_test = X_test[columns[:num_genes]].copy()
        else:
            if num_genes < len(top_genes):
                x_train = X_train[top_genes[:num_genes]].copy()
                x_test = X_test[top_genes[:num_genes]].copy()
            else:
                columns = list(X_train.columns)
                random.shuffle(columns)
                x_train = X_train[(top_genes[:] + columns)[:num_genes]].copy()
                x_test = X_test[(top_genes[:] + columns)[:num_genes]].copy()

        if 'MLP' not in model_name:
            neighbours_num = gene_graph.neighbors if num_genes>gene_graph.neighbors else num_genes-1
            gene_graph = OntologyGraph(neighbors=neighbours_num, embeddings_name=gene_graph.embeddings_name, randomize=gene_graph.randomize, gene_names=list(x_train.columns), relabel_genes=False)
            adj = sparse.csr_matrix(nx.to_numpy_matrix(gene_graph.nx_graph))
            print(adj.shape)
        else:
            adj=None
        



        if model_name == 'GCN':
            model = GCN(name="GCN_noemb_lay1_chan64_dropout_agg_hierarchical", #_lay3_chan64_emb32_dropout_agg_hierarchy", 
                                dropout=True, 
                                cuda=torch.cuda.is_available(),
                                num_layer=1,
                                prepool_extralayers=0,
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
        elif model_name == 'GCN16':
            model = GCN(name="GCN_noemb_lay1_chan16_dropout_agg_hierarchical", #_lay3_chan64_emb32_dropout_agg_hierarchy", 
                                dropout=True, 
                                cuda=torch.cuda.is_available(),
                                num_layer=1,
                                prepool_extralayers=0,
                                channels=16, 
                                embedding=2, # sdict["embedding"], 
                                aggregation="hierarchical",
                                lr=0.001,
                                num_epochs=100,
                                patience=30,
                                verbose=True,
                                seed=seed,
                                train_valid_split=0.8
                                )
        elif model_name == 'GCN2':
            model = GCN(name="GCN_noemb_lay1_chan64_dropout_agg_hierarchical_lr=0.0005", #_lay3_chan64_emb32_dropout_agg_hierarchy", 
                                dropout=True, 
                                cuda=torch.cuda.is_available(),
                                num_layer=1,
                                prepool_extralayers=0,
                                channels=64, 
                                embedding=64, # sdict["embedding"], 
                                aggregation="hierarchical",
                                lr=0.0005,
                                num_epochs=100,
                                patience=30,
                                verbose=True,
                                seed=seed,
                                train_valid_split=0.8
                                )
        elif model_name == 'GCN3':
            model = GCN(name="GCN_noemb_lay1_chan16_dropout_agg_hierarchical_prepool2", #_lay3_chan64_emb32_dropout_agg_hierarchy", 
                                dropout=True, 
                                cuda=torch.cuda.is_available(),
                                num_layer=1,
                                prepool_extralayers=2,
                                channels=16, 
                                embedding=16, # sdict["embedding"], 
                                aggregation="hierarchical",
                                lr=0.001,
                                num_epochs=100,
                                patience=30,
                                verbose=True,
                                seed=seed,
                                train_valid_split=0.8
                                )
        elif model_name == 'GCN_lay2':
            model = GCN(name="GCN_noemb_lay2_chan64_dropout_agg_hierarchical_prepool1", #_lay3_chan64_emb32_dropout_agg_hierarchy", 
                                dropout=True, 
                                cuda=torch.cuda.is_available(),
                                num_layer=2,
                                prepool_extralayers=1,
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
        elif model_name == 'GCN4':
            model = GCN(name="GCN_noemb_lay1_chan16_dropout_agg_hierarchical_prepool2_lr=0.0005", #_lay3_chan64_emb32_dropout_agg_hierarchy", 
                                dropout=True, 
                                cuda=torch.cuda.is_available(),
                                num_layer=1,
                                prepool_extralayers=2,
                                channels=16, 
                                embedding=16, # sdict["embedding"], 
                                aggregation="hierarchical",
                                lr=0.0005,
                                num_epochs=100,
                                patience=30,
                                verbose=True,
                                seed=seed,
                                train_valid_split=0.8
                                )
        elif model_name == 'GCN5':
            model = GCN(name="GCN_noemb_lay1_chan64_agg_hierarchical", #_lay3_chan64_emb32_dropout_agg_hierarchy", 
                                dropout=False, 
                                cuda=torch.cuda.is_available(),
                                num_layer=1,
                                prepool_extralayers=0,
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
        elif model_name == 'MLP1':
            model = MLP(name="MLP_lay1_chan64", cuda=cuda, dropout=True, num_layer=1, channels=64, train_valid_split=0.8, patience=30, lr=0.001)
        elif model_name == 'MLP2':
            model = MLP(name="MLP_lay2_chan64_lr.0.001", cuda=cuda, dropout=False, num_layer=2, channels=64, train_valid_split=0.8, patience=30, lr=0.001)
        elif model_name == 'MLP2.1':
            model = MLP(name="MLP_lay2_chan64_lr.0.001_dropout", cuda=cuda, dropout=True, num_layer=2, channels=64, train_valid_split=0.8, patience=30, lr=0.001)
        elif model_name == 'MLP3':
            model = MLP(name="MLP_lay2_chan16_lr.0.0001", cuda=cuda, dropout=False, num_layer=2, channels=16, train_valid_split=0.8, patience=30, lr=0.0001)
        elif model_name == 'MLP4':
            model = MLP(name="MLP_lay2_chan16_lr.0.0001_dropout", cuda=cuda, dropout=True, num_layer=2, channels=16, train_valid_split=0.8, patience=30, lr=0.0001)

        try:
            # print(x_train.shape,  y_train.shape, adj.shape)

            model.fit(x_train, y_train, adj=adj)

            with torch.no_grad():
                model.eval()
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
            print("Exception, x shape: ", x_train.shape)
            # print(str( gene in list(neighbors.nodes)))
            print(x_train.shape,  y_train.shape, adj.shape)
            print(e)
            logging.error(logging.traceback.format_exc())
            print("------------------------------------")

        # cleanup
        model.best_model = None  
        del model
        torch.cuda.empty_cache()