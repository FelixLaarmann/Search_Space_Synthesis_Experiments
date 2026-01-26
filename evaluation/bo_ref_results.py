from cl3s import (SpecificationBuilder, Constructor, Literal, Var, Group, DataGroup,
                  DerivationTree, SearchSpaceSynthesizer, BayesianOptimization,
                  WeisfeilerLehmanKernel, OptimizableHierarchicalWeisfeilerLehmanKernel)

import torch
import torch.nn as nn
import torch.optim as optim

from synthesis.utils import generate_data

import re
import time 

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

with open("results/20260126_131224/result_WL1.pkl", 'rb') as f: 
    data = dill.load(f)
print(data)


