#!/usr/bin/env python

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import meta_dataloader.TCGA

import models.mlp, models.gcn
import numpy as np
import logging
import data.gene_graphs
import collections
import sklearn.metrics
import sklearn.model_selection
import pandas as pd
import torch
from data.utils import record_result
import pickle
from scipy import sparse
import networkx as nx


import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-seed', type=int, default=0)
parser.add_argument('-ntrain', type=int, default=100)
parser.add_argument('-task', type=str, default="histological_type")
parser.add_argument('-study', type=str, default="LGG")
parser.add_argument('-graph', type=str, default="stringdb")
args = parser.parse_args()
print(args)

# tasks = meta_dataloader.TCGA.TCGAMeta(download=True, 
#                                       min_samples_per_class=10, 
#                                       gene_symbol_map_file="data/genenames_code_map_Feb2019.txt")


# # for taskid in tasks.task_ids:
# #     if "BRCA" in taskid:
# #         print(taskid)


# # clinical_M  PAM50Call_RNAseq
# task = meta_dataloader.TCGA.TCGATask((args.task, args.study), gene_symbol_map_file="data/genenames_code_map_Feb2019.txt")

# print(task.id)
# print(task._samples.shape)
# print(np.asarray(task._labels).shape)
# print(collections.Counter(task._labels))

# print(task._samples)

# df = pd.DataFrame(task._samples, columns = task.gene_ids)
# X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(df, 
#                                                                             task._labels, 
#                                                                             stratify=task._labels,
#                                                                             train_size=args.ntrain,
#                                                                             test_size=100,
#                                                                             shuffle=True,
#                                                                             random_state=args.seed
#                                                                              )
# X_test, X_valid, y_test, y_valid = sklearn.model_selection.train_test_split(X_test, 
#                                                                             y_test, 
#                                                                             stratify=y_test,
#                                                                             train_size=50,
#                                                                             test_size=50,
#                                                                             shuffle=True,
#                                                                             random_state=args.seed
#                                                                            )


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
else:
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

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(features, 
                                                                            target.to_numpy(), 
                                                                            stratify=target.to_numpy(),
                                                                            train_size=args.ntrain,
                                                                            test_size=800,
                                                                            shuffle=True,
                                                                            random_state=args.seed
                                                                             )
X_test, X_valid, y_test, y_valid = sklearn.model_selection.train_test_split(X_test, 
                                                                            y_test, 
                                                                            stratify=y_test,
                                                                            train_size=400,
                                                                            test_size=400,
                                                                            shuffle=True,
                                                                            random_state=args.seed
                                                                           )
                                                                           



# Setup the results dictionary
filename = "experiments/results/MB_hpsearch_" + str(args.ntrain) + "_" +args.task + ".pkl"
try:
    results = pickle.load(open(filename, "rb"))
    print("Loaded Checkpointed Results")
except FileNotFoundError as e:
    print(e)
    results = pd.DataFrame(columns=['auc', 'acc', 'gene', 'model', 'graph', 'num_genes', 'seed', 'train_size', 'lr', 'channels', 'embedding', 'num_layer', 'prepool_extralayers'])
    print("Created a New Results Dictionary")



import skopt, collections
from skopt.space import Real, Integer, Categorical






def doMLP():
    
    
    skopt_args = collections.OrderedDict()
    skopt_args["lr"]=Integer(2, 5)
    skopt_args["channels"]=Integer(4, 12)
    skopt_args["layers"]=Integer(1, 4)

    optimizer = skopt.Optimizer(dimensions=skopt_args.values(),
                                base_estimator="GP",
                                n_initial_points=3,
                                random_state=args.seed)
    print(skopt_args)

    best_valid_metric = 0
    test_for_best_valid_metric = 0
    best_config = None
    already_done = set()
    for i in range(10):
        suggestion = optimizer.ask()
        if str(suggestion) in already_done:
            continue
        already_done.add(str(suggestion))
        sdict = dict(zip(skopt_args.keys(),suggestion))
        sdict["lr"] = 10**float((-sdict["lr"]))
        sdict["channels"] = 2**sdict["channels"]

        model = models.mlp.MLP(name="MLP",
                               num_layer=sdict["layers"], 
                               channels=sdict["channels"], 
                               lr=sdict["lr"],
                               num_epochs=100,
                               patience=50,
                               cuda=torch.cuda.is_available(),
                               metric=sklearn.metrics.accuracy_score,
                               verbose=False,
                               seed=args.seed)

        model.fit(X_train, y_train)

        y_valid_pred = model.predict(X_valid)
        valid_metric = sklearn.metrics.accuracy_score(y_valid, np.argmax(y_valid_pred,axis=1))

        opt_results = optimizer.tell(suggestion, - valid_metric) 
        print(opt_results)

        #record metrics to write and plot
        if best_valid_metric < valid_metric:
            best_valid_metric = valid_metric
            best_config = sdict

            y_test_pred = model.predict(X_test)
            test_metric = sklearn.metrics.accuracy_score(y_test, np.argmax(y_test_pred,axis=1))
            test_for_best_valid_metric = test_metric

        print(i,"This result:",valid_metric, sdict)

        experiment = {
            "model": model.name,
            "graph": "",
            "num_genes":  len(list(X_train.columns)),
            "train_size": args.ntrain,
            "seed": args.seed,
            "acc": valid_metric,
            'lr': sdict["lr"], 
            'channels': sdict["channels"], 
            'embedding': 0, 
            'num_layer': sdict["layers"], 
            'prepool_extralayers': 0
        }

        global results
        results = record_result(results, experiment, filename)

    print("#Final Results", test_for_best_valid_metric, best_config)
    return test_metric, best_config



def doGGC():
    
    # if args.graph == "stringdb":
    #     graph = data.gene_graphs.StringDBGraph(datastore="./data")
    # elif args.graph == "genemania":
    #     graph = data.gene_graphs.GeneManiaGraph()
    # elif args.graph == "ontology":
    #     graph = data.gene_graphs.OntologyGraph(neighbors=30, embeddings_name='el')
    # else:
    #     print("unknown graph")
    #     sys.exit(1)

    gene_graphs = [
        OntologyGraph(neighbors=30, embeddings_name='dl2vec', randomize=False, gene_names=list(features.columns), relabel_genes=False),
        OntologyGraph(neighbors=30, embeddings_name='el', randomize=False, gene_names=list(features.columns), relabel_genes=False)]
        # data.gene_graphs.OntologyGraph(neighbors=n, embeddings_name=emb) 
        # for n in [30] #[500, 100, 30, 10] 
        # for emb in ['el', 'dl2vec', 'opa2vec', 'opa2vec_go']] + [
        #     data.gene_graphs.GeneManiaGraph(), 
        #     data.gene_graphs.StringDBGraph(datastore="./data")]
        # data.gene_graphs.OntologyGraph(neighbors=30, embeddings_name='el'), 
        # data.gene_graphs.OntologyGraph(neighbors=30, embeddings_name='el'), 
    # gene_graphs = [data.gene_graphs.OntologyGraph(neighbors=100, embeddings_name='el')]
        # data.gene_graphs.GeneManiaGraph(), 
        # data.gene_graphs.StringDBGraph(datastore="./data")]

    for graph in gene_graphs:

        adj = graph.adj()

        for num_genes in [1000, 16000]:
            for dropout in [True, False]: #, False]:
                import gc
                gc.collect()

                skopt_args = collections.OrderedDict()
                # skopt_args["lr"]=Integer(3, 4)
                skopt_args["channels"]=Integer(4, 9)
                # skopt_args["embedding"]=Integer(4, 5)
                skopt_args["num_layer"]=Integer(0, 4)
                skopt_args["prepool_extralayers"]=Integer(0, 3)

                optimizer = skopt.Optimizer(dimensions=skopt_args.values(),
                                            base_estimator="GP",
                                            n_initial_points=4,
                                            random_state=args.seed)
                print(skopt_args)



                best_valid_metric = 0
                test_for_best_valid_metric = 0
                best_config = None
                already_done = set()

                for i in range(15):
                    import gc
                    gc.collect()

                    suggestion = optimizer.ask()
                    
                    if str(suggestion) in already_done:
                        continue
                    already_done.add(str(suggestion))
                    sdict = dict(zip(skopt_args.keys(),suggestion))
                    sdict["lr"] = 0.001 #10**float((-sdict["lr"]))
                    sdict["channels"] = 2**sdict["channels"]
                    sdict["embedding"] = 2# 2**sdict["embedding"]
                    print(sdict)

                    gene = 'ESR1'
                    neighbors = graph.bfs_sample_neighbors(gene, num_genes*1.5)
                    intersection_nodes = np.intersect1d(features.columns, neighbors.nodes)

                    x_train = X_train[list(intersection_nodes)[:num_genes]].copy()
                    x_valid = X_valid[list(intersection_nodes)[:num_genes]].copy()
                    x_test = X_test[list(intersection_nodes)[:num_genes]].copy()

                    toremove = set(neighbors.nodes)
                    toremove = toremove.difference(intersection_nodes)
                    neighbors.remove_nodes_from(toremove)
                    adj = sparse.csr_matrix(nx.to_numpy_matrix(neighbors))

                    model = models.gcn.GCN(name="GCN_noemb" + ("_dropout" if dropout else ""), #_lay3_chan64_emb32_dropout_agg_hierarchy", 
                                        dropout=dropout, 
                                        cuda=torch.cuda.is_available(),
                                        num_layer=sdict["num_layer"],
                                        prepool_extralayers=sdict["prepool_extralayers"],
                                        channels=sdict["channels"], 
                                        embedding=sdict["embedding"], # sdict["embedding"], 
                                        aggregation="hierarchy",
                                        lr=sdict["lr"],
                                        num_epochs=100,
                                        patience=30,
                                        verbose=False,
                                        seed=args.seed
                                        )

                    try:
                        model.fit(x_train, y_train, adj)

                        with torch.no_grad():
                            model.eval()
                            y_valid_pred = model.predict(x_valid)
                            valid_metric = sklearn.metrics.accuracy_score(y_valid, np.argmax(y_valid_pred,axis=1))

                            opt_results = optimizer.tell(suggestion, - valid_metric) 

                            # #record metrics to write and plot
                            # if best_valid_metric < valid_metric:
                            #     best_valid_metric = valid_metric
                            #     print("best_valid_metric", best_valid_metric, sdict)
                            #     best_config = sdict

                            #     y_test_pred = model.predict(x_test)
                            #     test_metric = sklearn.metrics.accuracy_score(y_test, np.argmax(y_test_pred,axis=1))
                            #     test_for_best_valid_metric = test_metric


                            experiment = {
                                "model": model.name,
                                "graph": graph.graph_name,
                                "num_genes": num_genes,
                                "train_size": args.ntrain,
                                "seed": args.seed,
                                "acc": valid_metric,
                                'lr': sdict["lr"], 
                                'channels': sdict["channels"], 
                                'embedding': sdict["embedding"], 
                                'num_layer': sdict["num_layer"], 
                                'prepool_extralayers': sdict["prepool_extralayers"]
                            }
                            print(i, num_genes, "This result:",valid_metric, experiment)

                            global results
                            results = record_result(results, experiment, filename)

                    except Exception as e:
                        print(e)
                        logging.error(logging.traceback.format_exc())


                    # cleanup
                    model.best_model = None  
                    del model
                    torch.cuda.empty_cache()
                    


    print("#Final Results", test_for_best_valid_metric, best_config)
    return test_for_best_valid_metric, best_config


results_mlp = doMLP()
results_ggc = doGGC()


print("####GGC", args,results_ggc)
print("####MLP", args,results_mlp)
