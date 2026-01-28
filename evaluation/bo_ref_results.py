from cl3s import (SpecificationBuilder, Constructor, Literal, Var, Group, DataGroup,
                  DerivationTree, SearchSpaceSynthesizer, BayesianOptimization,
                  WeisfeilerLehmanKernel, OptimizableHierarchicalWeisfeilerLehmanKernel)

import torch
import torch.nn as nn
import torch.optim as optim

from synthesis.utils import generate_data

import re
import time 
import os 

from grakel.utils import graph_from_networkx

import networkx as nx

from synthesis.ode_1_repo import ODE_1_Repository

import dill
from pathlib import Path 
from datetime import datetime

def prepare_cls_data(x_p, repo):
    res = []
    for term in x_p:
        pickled_term = term.interpret(repo.pickle_algebra())
        res.append(pickled_term)
    return res 

def unpickle_cls_data(x_p, repo, synthesizer):
    res = []
    for idx, x_pickle in enumerate(x_p):
        target_ = repo.from_pickle(x_pickle)
        search_space_tmp = synthesizer.construct_search_space(target_)
        terms = list(search_space_tmp.enumerate_trees(target_, 10))
        # print(f'Num Terms: {len(list(terms))}')
        assert(len(list(terms)) == 1)
        res.append(list(terms)[0])
    return res 

repo = ODE_1_Repository(linear_feature_dimensions=[1, 2, 3, 4], constant_values=[0, 1, -1], learning_rate_values=[1e-2, 5e-3 ,1e-3],
                        n_epoch_values=[1000])

edge = (("swap", 0, 1), 1, 1)

target_solution = Constructor("Learner", Constructor("DAG",
                                                     Constructor("input", Literal(1))
                                                     & Constructor("output", Literal(1))
                                                     & Constructor("structure", Literal(
                                                         (
                                                            (
                                                                (ODE_1_Repository.Copy(2), 1, 2),
                                                            ),
                                                            (
                                                                edge,
                                                                (repo.Linear(1, 1, True), 1, 1),
                                                            ),
                                                            (
                                                                (repo.Linear(1, 1, True), 1, 1),
                                                                (repo.Tanh(), 1, 1),  
                                                            ),
                                                            (
                                                                (repo.Product(), 2, 1),
                                                            ),
                                                            (
                                                                (repo.Product(-1), 1, 1),
                                                            )
                                                         )
                                                     )))
                                & Constructor("Loss", Constructor("type", Literal(repo.MSEloss())))
                                & Constructor("Optimizer", Constructor("type", Literal(repo.Adam(1e-2))))
                                & Constructor("epochs", Literal(1000))
                    )

synthesizer = SearchSpaceSynthesizer(repo.specification(), {})
search_space = synthesizer.construct_search_space(target_solution).prune()

base = 'results/ode_1_bo'
ref = ['no_ref', 'ref']
result_files = set()
init_sample = [10, 50]
results = dict()

for r, i in zip(ref, init_sample): 
    if r not in results:
        results[r] = {}
    if i not in results[r]:
        results[r][i] = {}
    folder_name = f'{base}_{r}_{i}'
    folders = os.listdir(folder_name)
    for f_ in folders: 
        data_containing_folder = f'{folder_name}/{f_}'
        if os.path.isfile(data_containing_folder):
            continue
        result_data = os.listdir(data_containing_folder)
        for rd in result_data: 
            if 'result' in rd: 
                result_data = f'{data_containing_folder}/{rd}'
                print(f'Working on {data_containing_folder}/{rd}')
                with open(f'{data_containing_folder}/{rd}', 'rb') as f:
                    try:
                        data = dill.load(f)
                        data["best_tree"]  = unpickle_cls_data([data['best_tree']], repo, synthesizer)[0]
                        fn = os.path.splitext(rd)
                        if rd in results[r][i]:
                            results[r][i][rd]['tree'].append(data['best_tree'].interpret(repo.pretty_term_algebra()))
                            results[r][i][rd]['y'].append(min(data["y"]))
                        else: 
                            results[r][i][rd] = {}
                            results[r][i][rd]['tree'] = [data['best_tree'].interpret(repo.pretty_term_algebra())]
                            results[r][i][rd]['y'] = [min(data["y"])]
                        result_files.add(rd)
                    except (AttributeError, ValueError):
                        print(f'{data_containing_folder}/{rd} is faulty')
print(list(result_files))












# with open("results/ode_1_bo_no_ref_10/20260127_142345/result_WL1.pkl", 'rb') as f: 
#     data = dill.load(f)
# print(data.keys())
# data["best_tree"]  = unpickle_cls_data([data['best_tree']], repo, synthesizer)[0]
# print(data['best_tree'].interpret(repo.pretty_term_algebra()))
# print(f'y: {min(data["y"])}')
