from cl3s import (SpecificationBuilder, Constructor, Literal, Var, Group, DataGroup,
                  DerivationTree, SearchSpaceSynthesizer, BayesianOptimization,
                  WeisfeilerLehmanKernel, OptimizableHierarchicalWeisfeilerLehmanKernel)
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from synthesis.utils import generate_data
import re
import time
import numpy as np
from grakel.utils import graph_from_networkx
import networkx as nx
from synthesis.ode_1_repo import ODE_1_Repository
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.stats import pearsonr, spearmanr, kendalltau
from itertools import islice
import dill
import sys 
import time 

import warnings

from pathlib import Path
from datetime import datetime

warnings.filterwarnings('error')

EXPERIMENT_NUMBER = "S1"
starting = datetime.now().strftime("%Y%m%d_%H%M%S")
folder = f'results/{EXPERIMENT_NUMBER}/{starting}'
path = Path(folder)
path.mkdir(parents=True, exist_ok=True)



repo = ODE_1_Repository(linear_feature_dimensions=[1, 2, 3, 4], constant_values=[0, 1, -1], learning_rate_values=[1e-2, 5e-3 ,1e-3],
                        n_epoch_values=[1000])

#repo = ODE_1_Repository(linear_feature_dimensions=[1, 2], constant_values=[0, 1,],
#                        learning_rate_values=[1e-2], n_epoch_values=[10000])

edge = (("swap", 0, 1), 1, 1)


def parallel_edges(n):
    if n <= 0:
        raise ValueError("n must be positive")
    else:
        return (("swap", 0, n), n, n)
"""
target_trapezoid = Constructor("Learner", Constructor("DAG",
                                                      Constructor("input", Literal(3))
                                                      & Constructor("output", Literal(1))
                                                      & Constructor("structure", Literal(
                                                          (
                                                              (
                                                                  (repo_lte.Linear(1, 1, True), 1, 1),
                                                                  (repo_lte.Linear(1, 1, True), 1, 1),
                                                                  (repo_lte.Linear(1, 1, True), 1, 1)
                                                              ),  # left, split, right
                                                              (
                                                                  edge,
                                                                  (repo_lte.LTE(0), 1, 2),
                                                                  edge
                                                              ),  # left, gate, right
                                                              (
                                                                  (repo_lte.Product(), 2, 1),
                                                                  (repo_lte.Product(-1), 1, 1),
                                                                  edge
                                                              ),  # left_out, -gate, right
                                                              (
                                                                  edge,
                                                                  (repo_lte.Sum(1), 1, 1),
                                                                  edge
                                                              ),  # left_out, 1-gate, right
                                                              (
                                                                  edge,
                                                                  (repo_lte.Product(), 2, 1)
                                                              ),  # left_out, right_out
                                                              (
                                                                  (repo_lte.Sum(), 2, 1),
                                                              )
                                                          )
                                                      )))
                               & Constructor("Loss", Constructor("type", Literal(repo_lte.MSEloss())))
                               & Constructor("Optimizer", Constructor("type", Literal(repo_lte.Adam(1e-2))))
                               & Constructor("epochs", Literal(10000))
                               )
"""
""" 
None,
None,  # left, split, right
None,  # left, gate, right
None,  # left_out, -gate, right
None,  # left_out, 1-gate, right
None,  # left_out, right_out
None 
""" 
target_1 = Constructor("Learner", Constructor("DAG",
                                                          Constructor("input", Literal(1))
                                                          & Constructor("output", Literal(1))
                                                          & Constructor("structure", Literal(
                                                              (
                                                                  None,
                                                                  None,  # left, split, right
                                                                  None,  # left, gate, right
                                                                  None,  # left, gate, right
                                                                  None,  # left_out, -gate, right
                                                                  #None,  # left_out, 1-gate, right
                                                                  #None,  # left_out, right_out
                                                                  #None
                                                              )
                                                          )))
                                   & Constructor("Loss", Constructor("type", Literal(None)))
                                   & Constructor("Optimizer", Constructor("type", Literal(None)))
                                   & Constructor("epochs", Literal(1000))
                                   )


target_2 = Constructor("Learner", Constructor("DAG",
                                                            Constructor("input", Literal(1))
                                                            & Constructor("output", Literal(1))
                                                            & Constructor("structure", Literal(
                                                                (
                                                                    (
                                                                        None,
                                                                    ),
                                                                    (
                                                                        None,
                                                                        None,
                                                                    ), 
                                                                    (
                                                                        None,
                                                                        None,
                                                                    ), 
                                                                    (
                                                                        None,
                                                                    ), 
                                                                    (
                                                                        None,
                                                                    ), 
                                                                )
                                                            )))
                                     & Constructor("Loss", Constructor("type", Literal(None)))
                                     & Constructor("Optimizer", Constructor("type", Literal(repo.Adam(1e-2))))
                                     & Constructor("epochs", Literal(10000))
                                     )

target_3 = Constructor("Learner", Constructor("DAG",
                                                                Constructor("input", Literal(1))
                                                                & Constructor("output", Literal(1))
                                                                & Constructor("structure", Literal(
                                                                    (
                                                                        (
                                                                            (None, 1, 2),
                                                                        ),
                                                                        (
                                                                            (None, 1, 1),
                                                                            (None, 1, 1),
                                                                        ),
                                                                        (
                                                                            edge,
                                                                            (None, 1, 1),
                                                                        ),
                                                                        (
                                                                            (None, 2, 1),
                                                                        ), 
                                                                        (
                                                                            (None, 1, 1),
                                                                        ), 
                                                                    )
                                                                )))
                                         & Constructor("Loss", Constructor("type", Literal(None)))
                                         & Constructor("Optimizer", Constructor("type", Literal(repo.Adam(1e-2))))
                                         & Constructor("epochs", Literal(10000))
                                         )

def to_grakel_graph_1(t):
    edgelist = t.interpret(repo.edgelist_algebra())

    G = nx.MultiDiGraph()
    G.add_edges_from(edgelist)

    relabel = {n: "Activation" if ("Sigmoid" in n or "ReLu" in n or "Tanh" in n) else "Node"
               for n in G.nodes()}

    #relabel = {n: "Node"
    #           for n in G.nodes()}

    for n in G.nodes():
        G.nodes[n]['symbol'] = relabel[n]

    gk_graph = graph_from_networkx([G.to_undirected()], node_labels_tag='symbol')

    return gk_graph

def to_grakel_graph_2(t):
    edgelist = t.interpret(repo.edgelist_algebra())

    G = nx.MultiDiGraph()
    G.add_edges_from(edgelist)

    relabel = {n: "Linear" if "Linear" in n else "Sigmoid" if "Sigmoid" in n else "ReLu" if "ReLu" in n else
                    "Tanh" if "Tanh" in n else
                    "Sum" if "Sum" in n else
                    "Product" if "Product" in n else
                    "Copy" if "Copy" in n else
                    "Node"
               for n in G.nodes()}

    for n in G.nodes():
        G.nodes[n]['symbol'] = relabel[n]

    gk_graph = graph_from_networkx([G.to_undirected()], node_labels_tag='symbol')

    return gk_graph

def to_grakel_graph_3(t):
    edgelist = t.interpret(repo.edgelist_algebra())

    G = nx.MultiDiGraph()
    G.add_edges_from(edgelist)

    relabel = {n: re.sub("[)][(][-]*[0-9]*[.][0-9]*[,]\s[-]*[0-9]*[.][0-9]*[)]", ")", n)
               for n in G.nodes()}

    for n in G.nodes():
        G.nodes[n]['symbol'] = relabel[n]

    gk_graph = graph_from_networkx([G.to_undirected()], node_labels_tag='symbol')

    return gk_graph

if __name__ == "__main__":

    plot_resolution = 40

    sample_size = 500

    test_size = int(sample_size // 5)

    train_size = sample_size - test_size

    target = target_1

    synthesizer = SearchSpaceSynthesizer(repo.specification(), {})
    start = time.time()
    try:
        search_space = synthesizer.construct_search_space(target).prune()
    except MemoryError:
        print(f'Out of Memory Error')
    end = time.time() 

    print(f"finished synthesis: {sys.getsizeof(search_space)} bytes")
    print(f'Elapsed Time: {end - start}')
    with open(f'{folder}/search_space_{EXPERIMENT_NUMBER}.pkl', 'wb') as f:
        dill.dump(search_space, f)

    # test = search_space.enumerate_trees(target, 10)

    # test_list = list(test)
    # print(f"Number of trees found: {len(test_list)}")

    kernel1 = WeisfeilerLehmanKernel(n_iter=1, to_grakel_graph=to_grakel_graph_1)
    kernel2 = WeisfeilerLehmanKernel(n_iter=1, to_grakel_graph=to_grakel_graph_2)
    kernel3 = WeisfeilerLehmanKernel(n_iter=1, to_grakel_graph=to_grakel_graph_3)
    hkernel = OptimizableHierarchicalWeisfeilerLehmanKernel(to_grakel_graph1=to_grakel_graph_1,
                                                            to_grakel_graph2=to_grakel_graph_2,
                                                            to_grakel_graph3=to_grakel_graph_3,
                                                            weight1=0.4, weight2=0.3, weight3=0.3,
                                                            n_iter1=1, n_iter2=1, n_iter3=1)

    gp1 = GaussianProcessRegressor(kernel=kernel1, optimizer=None, normalize_y=False)
    gp2 = GaussianProcessRegressor(kernel=kernel2, optimizer=None, normalize_y=False)
    gp3 = GaussianProcessRegressor(kernel=kernel3, optimizer=None, normalize_y=False)
    # the hierarchical kernel has hyperparameters, which can be optimized while fitting the GP
    gp_h = GaussianProcessRegressor(kernel=hkernel, optimizer=hkernel.optimize_hyperparameter, n_restarts_optimizer=2,
                                    normalize_y=False)

    # Load pre generated data for the training
    data = torch.load('../data/ode1_dataset.pth')
    x_data_loaded = data['x_train']
    y_data_loaded = data['y_train']
    x_test_data_loaded = data['x_test']
    y_test_data_loaded = data['y_test']

    def f_obj(t):
        learner = t.interpret(repo.pytorch_function_algebra())
        return learner(x_data_loaded, y_data_loaded, x_test_data_loaded, y_test_data_loaded)

    #terms = search_space.sample(sample_size, target)
    next = search_space.sample_tree(target)
    terms = []
    while len(terms) < sample_size:
        print(len(terms))
        is_duplicate = False
        for tree in terms:
            k = kernel1._f(next, tree)  # kernel1 should be enough
            if k > 0.99:  # almost identical
                is_duplicate = True
                break
        if not is_duplicate:
            terms.append(next)
        next = search_space.sample_tree(target)

    
    # test = search_space.enumerate_trees(target, 10)

    # print(f"Number of trees found: {len(test_list)}")
    
    x_gp = terms #list(terms)
    print(f'Number of terms: {len(x_gp)}')
    y_gp = [f_obj(t) for t in x_gp]

    print("duplicates in data:")
    print("X: ", len(x_gp) - len(set(x_gp)))
    print("Y: ", len(y_gp) - len(set(y_gp)))

    print("X should not have any duplicates!")
    print("If Y has duplicates, either the objective function is not injective or its a rounding error.")

    x_gp_test_robust = np.array(x_gp[train_size:])
    y_gp_test_robust = np.array(y_gp[train_size:])

    next_test = search_space.sample_tree(target)
    terms = []
    while len(terms) < test_size:
        print(len(terms))
        is_duplicate = False
        for tree in terms:
            k = kernel1._f(next_test, tree)  # kernel1 should be enough
            if k > 0.8:  # sample more different test set
                is_duplicate = True
                break
        if not is_duplicate:
            terms.append(next_test)
        next_test = search_space.sample_tree(target)

    x_gp_test_different= np.array(terms)
    y_gp_test_different = np.array([f_obj(t) for t in terms])

    last_test = search_space.sample(test_size, target)
    x_gp_test_redundancy = np.array(last_test)
    y_gp_test_redundancy = np.array([f_obj(t) for t in last_test])

    slice_size = int(train_size // plot_resolution)
    #print(f'slice size: {slice_size}')

    x_gp_train = [np.array(x_gp[i:i + slice_size]) for i in range(0, train_size, slice_size)] #[x_gp_train_1, x_gp_train_2, x_gp_train_3, x_gp_train_4, x_gp_train_5, x_gp_train_6, x_gp_train_7, x_gp_train_8]
    y_gp_train = [np.array(y_gp[i:i + slice_size]) for i in range(0, train_size, slice_size)] #[y_gp_train_1, y_gp_train_2, y_gp_train_3, y_gp_train_4, y_gp_train_5, y_gp_train_6, y_gp_train_7, y_gp_train_8]
    y_preds_gp1 = []
    y_sigmas_gp1 = []
    y_preds_gp2 = []
    y_sigmas_gp2 = []
    y_preds_gp3 = []
    y_sigmas_gp3 = []
    y_preds_gp_h = []
    y_sigmas_gp_h = []

    pears_gp1_robust = []
    kts_gp1_robust = []
    pears_gp1_difference = []
    kts_gp1_difference = []
    pears_gp1_redundancy = []
    kts_gp1_redundancy = []
    pears_gp2_robust = []
    kts_gp2_robust = []
    pears_gp2_difference = []
    kts_gp2_difference = []
    pears_gp2_redundancy = []
    kts_gp2_redundancy = []
    pears_gp3_robust = []
    kts_gp3_robust = []
    pears_gp3_difference = []
    kts_gp3_difference = []
    pears_gp3_redundancy = []
    kts_gp3_redundancy = []
    pears_gp_h_robust = []
    kts_gp_h_robust = []
    pears_gp_h_difference = []
    kts_gp_h_difference = []
    pears_gp_h_redundancy = []
    kts_gp_h_redundancy = []

    #print("data generated")

    x_trained = []
    y_trained = []

    for idx, (x_gp_i, y_gp_i) in enumerate(zip(x_gp_train, y_gp_train)):

        x_trained = x_trained + list(x_gp_i)
        y_trained = y_trained + list(y_gp_i)

        K1 = kernel1(x_trained)
        D1 = kernel1.diag(x_trained)

        plt.figure(figsize=(8, 5))
        plt.imshow(np.diag(D1 ** -0.5).dot(K1).dot(np.diag(D1 ** -0.5)))
        plt.xticks(np.arange(len(x_trained)), range(1, len(x_trained) + 1))
        plt.yticks(np.arange(len(x_trained)), range(1, len(x_trained) + 1))
        plt.title("Term similarity under the kernel1")
        #print(f'Doing: {folder}/term_sim_k1_{idx}_{EXPERIMENT_NUMBER}.png')
        plt.savefig(f'{folder}/term_sim_k1_{idx}_{EXPERIMENT_NUMBER}.png')
        plt.savefig(f'{folder}/term_sim_k1_{idx}_{EXPERIMENT_NUMBER}.pdf')
        plt.close()
        try:
            gp1.fit(x_trained, y_trained)
            y_pred_next, sigma_next = gp1.predict(x_gp_test_robust, return_std=True)
            y_pred_next_different, sigma_next_different = gp1.predict(x_gp_test_different, return_std=True)
            y_pred_next_redundancy, sigma_next_redundancy = gp1.predict(x_gp_test_redundancy, return_std=True)
        except Warning as e: 
            print("Abort when warning during GP fitting/prediction:")
            print(f"Look at {folder}/term_sim_k1_{idx}_{EXPERIMENT_NUMBER}.png")
            raise 


        y_preds_gp1.append(y_pred_next)
        y_sigmas_gp1.append(sigma_next)
        pears_gp1_robust.append(pearsonr(y_gp_test_robust, np.nan_to_num(y_pred_next))[0])
        kts_gp1_robust.append(kendalltau(y_gp_test_robust, np.nan_to_num(y_pred_next))[0])

        pears_gp1_difference.append(pearsonr(y_gp_test_different, np.nan_to_num(y_pred_next_different))[0])
        kts_gp1_difference.append(kendalltau(y_gp_test_different, np.nan_to_num(y_pred_next_different))[0])

        pears_gp1_redundancy.append(pearsonr(y_gp_test_redundancy, np.nan_to_num(y_pred_next_redundancy))[0])
        kts_gp1_redundancy.append(kendalltau(y_gp_test_redundancy, np.nan_to_num(y_pred_next_redundancy))[0])

        K2 = kernel2(x_trained)
        D2 = kernel2.diag(x_trained)

        plt.figure(figsize=(8, 5))
        plt.imshow(np.diag(D2 ** -0.5).dot(K2).dot(np.diag(D2 ** -0.5)))
        plt.xticks(np.arange(len(x_trained)), range(1, len(x_trained) + 1))
        plt.yticks(np.arange(len(x_trained)), range(1, len(x_trained) + 1))
        plt.title("Term similarity under the kernel2")
        plt.savefig(f'{folder}/term_sim_k2_{idx}_{EXPERIMENT_NUMBER}.png')
        plt.savefig(f'{folder}/term_sim_k2_{idx}_{EXPERIMENT_NUMBER}.pdf')
        plt.close()

        try:

            gp2.fit(x_trained, y_trained)

            y_pred_next, sigma_next = gp2.predict(x_gp_test_robust, return_std=True)
            y_pred_next_different, sigma_next_different = gp2.predict(x_gp_test_different, return_std=True)
            y_pred_next_redundancy, sigma_next_redundancy = gp2.predict(x_gp_test_redundancy, return_std=True)
        except Warning as e:
            print("Abort when warning during GP fitting/prediction:")
            print(f"Look at {folder}/term_sim_k2_{idx}_{EXPERIMENT_NUMBER}.png")
            raise

        y_preds_gp2.append(y_pred_next)
        y_sigmas_gp2.append(sigma_next)
        pears_gp2_robust.append(pearsonr(y_gp_test_robust, np.nan_to_num(y_pred_next))[0])
        kts_gp2_robust.append(kendalltau(y_gp_test_robust, np.nan_to_num(y_pred_next))[0])

        pears_gp2_difference.append(pearsonr(y_gp_test_different, np.nan_to_num(y_pred_next_different))[0])
        kts_gp2_difference.append(kendalltau(y_gp_test_different, np.nan_to_num(y_pred_next_different))[0])

        pears_gp2_redundancy.append(pearsonr(y_gp_test_redundancy, np.nan_to_num(y_pred_next_redundancy))[0])
        kts_gp2_redundancy.append(kendalltau(y_gp_test_redundancy, np.nan_to_num(y_pred_next_redundancy))[0])

        K3 = kernel3(x_trained)
        D3 = kernel3.diag(x_trained)

        plt.figure(figsize=(8, 5))
        plt.imshow(np.diag(D3 ** -0.5).dot(K3).dot(np.diag(D3 ** -0.5)))
        plt.xticks(np.arange(len(x_trained)), range(1, len(x_trained) + 1))
        plt.yticks(np.arange(len(x_trained)), range(1, len(x_trained) + 1))
        plt.title("Term similarity under the kernel3")
        plt.savefig(f'{folder}/term_sim_k3_{idx}_{EXPERIMENT_NUMBER}.png')
        plt.savefig(f'{folder}/term_sim_k3_{idx}_{EXPERIMENT_NUMBER}.pdf')
        plt.close()
        try:

            gp3.fit(x_trained, y_trained)

            y_pred_next, sigma_next = gp3.predict(x_gp_test_robust, return_std=True)
            y_pred_next_different, sigma_next_different = gp3.predict(x_gp_test_different, return_std=True)
            y_pred_next_redundancy, sigma_next_redundancy = gp3.predict(x_gp_test_redundancy, return_std=True)
        except Warning as e:
            print("Abort when warning during GP fitting/prediction:")
            print(f"Look at {folder}/term_sim_k3_{idx}_{EXPERIMENT_NUMBER}.png")
            raise

        y_preds_gp3.append(y_pred_next)
        y_sigmas_gp3.append(sigma_next)
        pears_gp3_robust.append(pearsonr(y_gp_test_robust, np.nan_to_num(y_pred_next))[0])
        kts_gp3_robust.append(kendalltau(y_gp_test_robust, np.nan_to_num(y_pred_next))[0])

        pears_gp3_difference.append(pearsonr(y_gp_test_different, np.nan_to_num(y_pred_next_different))[0])
        kts_gp3_difference.append(kendalltau(y_gp_test_different, np.nan_to_num(y_pred_next_different))[0])

        pears_gp3_redundancy.append(pearsonr(y_gp_test_redundancy, np.nan_to_num(y_pred_next_redundancy))[0])
        kts_gp3_redundancy.append(kendalltau(y_gp_test_redundancy, np.nan_to_num(y_pred_next_redundancy))[0])


        # Hierarchical kernel with initial hyperparameters
        K_h = hkernel(x_trained)
        D_h = hkernel.diag(x_trained)

        plt.figure(figsize=(8, 5))
        plt.imshow(np.diag(D3 ** -0.5).dot(K3).dot(np.diag(D3 ** -0.5)))
        plt.xticks(np.arange(len(x_trained)), range(1, len(x_trained) + 1))
        plt.yticks(np.arange(len(x_trained)), range(1, len(x_trained) + 1))
        plt.title("Term similarity under the unfitted hierarchical kernel")
        plt.savefig(f'{folder}/term_sim_hk_{idx}_{EXPERIMENT_NUMBER}.png')
        plt.savefig(f'{folder}/term_sim_hk_{idx}_{EXPERIMENT_NUMBER}.pdf')
        plt.close()

        # Hierarchical kernel with fitted hyperparameters from the last iteration
        K_h_fitted = gp_h.kernel(x_trained)
        D_h_fitted = gp_h.kernel.diag(x_trained)

        hyperparameters_fitted = gp_h.kernel.hyperparameters
        """
        For the first iteration, hyperparameters are the initial ones and K_h_fitted, D_h_fitted are the same as 
        K_h, D_h.
        For the following iterations, hyperparameters are the ones optimized from the last iteration and the 
        kernel matrices will (hopefully) differ.
        """

        ################## Save Kernels
        np.savez_compressed(f'{folder}/kernels_{idx}_{EXPERIMENT_NUMBER}.npz', k1=K1, d1=D1, k2=K2, d2=D2,
                            k3=K3, d3=D3, kh=K_h, dh=D_h, khfitted=K_h_fitted, dhfitted=D_h_fitted)
                            #hps=hyperparameters_fitted)

        ##################

        plt.figure(figsize=(8, 5))
        plt.imshow(np.diag(D3 ** -0.5).dot(K3).dot(np.diag(D3 ** -0.5)))
        plt.xticks(np.arange(len(x_trained)), range(1, len(x_trained) + 1))
        plt.yticks(np.arange(len(x_trained)), range(1, len(x_trained) + 1))
        plt.title("Term similarity under the fitted hierarchical kernel from the last iteration")
        plt.savefig(f'{folder}/term_sim_hkf_{idx}_{EXPERIMENT_NUMBER}.png')
        plt.savefig(f'{folder}/term_sim_hkf_{idx}_{EXPERIMENT_NUMBER}.pdf')
        plt.close()

        try:
            gp_h.fit(x_trained, y_trained)

            y_pred_next, sigma_next = gp_h.predict(x_gp_test_robust, return_std=True)
            y_pred_next_different = gp_h.predict(x_gp_test_different, return_std=True)
            y_pred_next_redundancy = gp_h.predict(x_gp_test_redundancy, return_std=True)
        except Warning as e:
            print("Abort when warning during GP fitting/prediction:")
            print(f"Look at {folder}/term_sim_hk_{idx}_{EXPERIMENT_NUMBER}.png")
            raise

        y_preds_gp_h.append(y_pred_next)
        y_sigmas_gp_h.append(sigma_next)
        pears_gp_h_robust.append(pearsonr(y_gp_test_robust, np.nan_to_num(y_pred_next))[0])
        kts_gp_h_robust.append(kendalltau(y_gp_test_robust, np.nan_to_num(y_pred_next))[0])

        pears_gp_h_difference.append(pearsonr(y_gp_test_different, np.nan_to_num(y_pred_next_different))[0])
        kts_gp_h_difference.append(kendalltau(y_gp_test_different, np.nan_to_num(y_pred_next_different))[0])

        pears_gp_h_redundancy.append(pearsonr(y_gp_test_redundancy, np.nan_to_num(y_pred_next_redundancy))[0])
        kts_gp_h_redundancy.append(kendalltau(y_gp_test_redundancy, np.nan_to_num(y_pred_next_redundancy))[0])

    plt.plot(range(slice_size, train_size + slice_size, slice_size), kts_gp1_robust, linestyle="dotted")
    plt.xlabel("# of samples")
    plt.ylabel("tau")
    _ = plt.title("Kendall Tau correlation for GP with kernel1 and robust test data")
    plt.savefig(f'{folder}/ktau_k1_{EXPERIMENT_NUMBER}.png')
    plt.savefig(f'{folder}/ktau_k1_{EXPERIMENT_NUMBER}.pdf')
    plt.close()

    plt.plot(range(slice_size, train_size + slice_size, slice_size), pears_gp1_robust, linestyle="dotted")
    plt.xlabel("# of samples")
    plt.ylabel("p")
    _ = plt.title("Pearson correlation for GP with kernel1 and robust test data")
    plt.savefig(f'{folder}/pc_k1_{EXPERIMENT_NUMBER}.png')
    plt.savefig(f'{folder}/pc_k1_{EXPERIMENT_NUMBER}.pdf')
    plt.close()

    plt.plot(range(slice_size, train_size + slice_size, slice_size), kts_gp2_robust, linestyle="dotted")
    plt.xlabel("# of samples")
    plt.ylabel("tau")
    _ = plt.title("Kendall Tau correlation for GP with kernel2 and robust test data")
    plt.savefig(f'{folder}/ktau_k2_{EXPERIMENT_NUMBER}.png')
    plt.savefig(f'{folder}/ktau_k2_{EXPERIMENT_NUMBER}.pdf')
    plt.close()

    plt.plot(range(slice_size, train_size + slice_size, slice_size), pears_gp2_robust, linestyle="dotted")
    plt.xlabel("# of samples")
    plt.ylabel("p")
    _ = plt.title("Pearson correlation for GP with kernel2 and robust test data")
    plt.savefig(f'{folder}/pc_k2_{EXPERIMENT_NUMBER}.png')
    plt.savefig(f'{folder}/pc_k2_{EXPERIMENT_NUMBER}.pdf')
    plt.close()

    plt.plot(range(slice_size, train_size + slice_size, slice_size), kts_gp3_robust, linestyle="dotted")
    plt.xlabel("# of samples")
    plt.ylabel("tau")
    _ = plt.title("Kendall Tau correlation for GP with kernel3 and robust test data")
    plt.savefig(f'{folder}/ktau_k3_{EXPERIMENT_NUMBER}.png')
    plt.savefig(f'{folder}/ktau_k3_{EXPERIMENT_NUMBER}.pdf')
    plt.close()

    plt.plot(range(slice_size, train_size + slice_size, slice_size), pears_gp3_robust, linestyle="dotted")
    plt.xlabel("# of samples")
    plt.ylabel("p")
    _ = plt.title("Pearson correlation for GP with kernel3 and robust test data")
    plt.savefig(f'{folder}/pc_k3_{EXPERIMENT_NUMBER}.png')
    plt.savefig(f'{folder}/pc_k3_{EXPERIMENT_NUMBER}.pdf')
    plt.close()

    plt.plot(range(slice_size, train_size + slice_size, slice_size), kts_gp_h_robust, linestyle="dotted")
    plt.xlabel("# of samples")
    plt.ylabel("tau")
    _ = plt.title("Kendall Tau correlation for GP with hierarchical kernel (with HPO) and robust test data")
    plt.savefig(f'{folder}/ktau_hk_{EXPERIMENT_NUMBER}.png')
    plt.savefig(f'{folder}/ktau_hk_{EXPERIMENT_NUMBER}.pdf')
    plt.close()

    plt.plot(range(slice_size, train_size + slice_size, slice_size), pears_gp_h_robust, linestyle="dotted")
    plt.xlabel("# of samples")
    plt.ylabel("p")
    _ = plt.title("Pearson correlation for GP with hierarchical kernel (with HPO) and robust test data")
    plt.savefig(f'{folder}/pc_hk_{EXPERIMENT_NUMBER}.png')
    plt.savefig(f'{folder}/pc_hk_{EXPERIMENT_NUMBER}.pdf')
    plt.close()

    plt.plot(range(slice_size, train_size + slice_size, slice_size), kts_gp1_difference, linestyle="dotted")
    plt.xlabel("# of samples")
    plt.ylabel("tau")
    _ = plt.title("Kendall Tau correlation for GP with kernel1 and test data with high variance")
    plt.savefig(f'{folder}/ktau_k1_{EXPERIMENT_NUMBER}.png')
    plt.savefig(f'{folder}/ktau_k1_{EXPERIMENT_NUMBER}.pdf')
    plt.close()

    plt.plot(range(slice_size, train_size + slice_size, slice_size), pears_gp1_difference, linestyle="dotted")
    plt.xlabel("# of samples")
    plt.ylabel("p")
    _ = plt.title("Pearson correlation for GP with kernel1 and test data with high variance")
    plt.savefig(f'{folder}/pc_k1_{EXPERIMENT_NUMBER}.png')
    plt.savefig(f'{folder}/pc_k1_{EXPERIMENT_NUMBER}.pdf')
    plt.close()

    plt.plot(range(slice_size, train_size + slice_size, slice_size), kts_gp2_difference, linestyle="dotted")
    plt.xlabel("# of samples")
    plt.ylabel("tau")
    _ = plt.title("Kendall Tau correlation for GP with kernel2 and test data with high variance")
    plt.savefig(f'{folder}/ktau_k2_{EXPERIMENT_NUMBER}.png')
    plt.savefig(f'{folder}/ktau_k2_{EXPERIMENT_NUMBER}.pdf')
    plt.close()

    plt.plot(range(slice_size, train_size + slice_size, slice_size), pears_gp2_difference, linestyle="dotted")
    plt.xlabel("# of samples")
    plt.ylabel("p")
    _ = plt.title("Pearson correlation for GP with kernel2 and test data with high variance")
    plt.savefig(f'{folder}/pc_k2_{EXPERIMENT_NUMBER}.png')
    plt.savefig(f'{folder}/pc_k2_{EXPERIMENT_NUMBER}.pdf')
    plt.close()

    plt.plot(range(slice_size, train_size + slice_size, slice_size), kts_gp3_difference, linestyle="dotted")
    plt.xlabel("# of samples")
    plt.ylabel("tau")
    _ = plt.title("Kendall Tau correlation for GP with kernel3 and test data with high variance")
    plt.savefig(f'{folder}/ktau_k3_{EXPERIMENT_NUMBER}.png')
    plt.savefig(f'{folder}/ktau_k3_{EXPERIMENT_NUMBER}.pdf')
    plt.close()

    plt.plot(range(slice_size, train_size + slice_size, slice_size), pears_gp3_difference, linestyle="dotted")
    plt.xlabel("# of samples")
    plt.ylabel("p")
    _ = plt.title("Pearson correlation for GP with kernel3 and test data with high variance")
    plt.savefig(f'{folder}/pc_k3_{EXPERIMENT_NUMBER}.png')
    plt.savefig(f'{folder}/pc_k3_{EXPERIMENT_NUMBER}.pdf')
    plt.close()

    plt.plot(range(slice_size, train_size + slice_size, slice_size), kts_gp_h_difference, linestyle="dotted")
    plt.xlabel("# of samples")
    plt.ylabel("tau")
    _ = plt.title("Kendall Tau correlation for GP with hierarchical kernel (with HPO) and test data with high variance")
    plt.savefig(f'{folder}/ktau_hk_{EXPERIMENT_NUMBER}.png')
    plt.savefig(f'{folder}/ktau_hk_{EXPERIMENT_NUMBER}.pdf')
    plt.close()

    plt.plot(range(slice_size, train_size + slice_size, slice_size), pears_gp_h_difference, linestyle="dotted")
    plt.xlabel("# of samples")
    plt.ylabel("p")
    _ = plt.title("Pearson correlation for GP with hierarchical kernel (with HPO) and test data with high variance")
    plt.savefig(f'{folder}/pc_hk_{EXPERIMENT_NUMBER}.png')
    plt.savefig(f'{folder}/pc_hk_{EXPERIMENT_NUMBER}.pdf')
    plt.close()

    plt.plot(range(slice_size, train_size + slice_size, slice_size), kts_gp1_redundancy, linestyle="dotted")
    plt.xlabel("# of samples")
    plt.ylabel("tau")
    _ = plt.title("Kendall Tau correlation for GP with kernel1 and possibly redundant test data")
    plt.savefig(f'{folder}/ktau_k1_{EXPERIMENT_NUMBER}.png')
    plt.savefig(f'{folder}/ktau_k1_{EXPERIMENT_NUMBER}.pdf')
    plt.close()

    plt.plot(range(slice_size, train_size + slice_size, slice_size), pears_gp1_redundancy, linestyle="dotted")
    plt.xlabel("# of samples")
    plt.ylabel("p")
    _ = plt.title("Pearson correlation for GP with kernel1 and possibly redundant test data")
    plt.savefig(f'{folder}/pc_k1_{EXPERIMENT_NUMBER}.png')
    plt.savefig(f'{folder}/pc_k1_{EXPERIMENT_NUMBER}.pdf')
    plt.close()

    plt.plot(range(slice_size, train_size + slice_size, slice_size), kts_gp2_redundancy, linestyle="dotted")
    plt.xlabel("# of samples")
    plt.ylabel("tau")
    _ = plt.title("Kendall Tau correlation for GP with kernel2 and possibly redundant test data")
    plt.savefig(f'{folder}/ktau_k2_{EXPERIMENT_NUMBER}.png')
    plt.savefig(f'{folder}/ktau_k2_{EXPERIMENT_NUMBER}.pdf')
    plt.close()

    plt.plot(range(slice_size, train_size + slice_size, slice_size), pears_gp2_redundancy, linestyle="dotted")
    plt.xlabel("# of samples")
    plt.ylabel("p")
    _ = plt.title("Pearson correlation for GP with kernel2 and possibly redundant test data")
    plt.savefig(f'{folder}/pc_k2_{EXPERIMENT_NUMBER}.png')
    plt.savefig(f'{folder}/pc_k2_{EXPERIMENT_NUMBER}.pdf')
    plt.close()

    plt.plot(range(slice_size, train_size + slice_size, slice_size), kts_gp3_redundancy, linestyle="dotted")
    plt.xlabel("# of samples")
    plt.ylabel("tau")
    _ = plt.title("Kendall Tau correlation for GP with kernel3 and possibly redundant test data")
    plt.savefig(f'{folder}/ktau_k3_{EXPERIMENT_NUMBER}.png')
    plt.savefig(f'{folder}/ktau_k3_{EXPERIMENT_NUMBER}.pdf')
    plt.close()

    plt.plot(range(slice_size, train_size + slice_size, slice_size), pears_gp3_redundancy, linestyle="dotted")
    plt.xlabel("# of samples")
    plt.ylabel("p")
    _ = plt.title("Pearson correlation for GP with kernel3 and possibly redundant test data")
    plt.savefig(f'{folder}/pc_k3_{EXPERIMENT_NUMBER}.png')
    plt.savefig(f'{folder}/pc_k3_{EXPERIMENT_NUMBER}.pdf')
    plt.close()

    plt.plot(range(slice_size, train_size + slice_size, slice_size), kts_gp_h_redundancy, linestyle="dotted")
    plt.xlabel("# of samples")
    plt.ylabel("tau")
    _ = plt.title("Kendall Tau correlation for GP with hierarchical kernel (with HPO) and possibly redundant test data")
    plt.savefig(f'{folder}/ktau_hk_{EXPERIMENT_NUMBER}.png')
    plt.savefig(f'{folder}/ktau_hk_{EXPERIMENT_NUMBER}.pdf')
    plt.close()

    plt.plot(range(slice_size, train_size + slice_size, slice_size), pears_gp_h_redundancy, linestyle="dotted")
    plt.xlabel("# of samples")
    plt.ylabel("p")
    _ = plt.title("Pearson correlation for GP with hierarchical kernel (with HPO) and possibly redundant test data")
    plt.savefig(f'{folder}/pc_hk_{EXPERIMENT_NUMBER}.png')
    plt.savefig(f'{folder}/pc_hk_{EXPERIMENT_NUMBER}.pdf')
    plt.close()



    # Save data
    regression_data = {
        'x_gp_train' : x_gp_train,
        'y_gp_train' : y_gp_train,
        'y_preds_gp1': y_preds_gp1,
        'y_sigmas_gp1': y_sigmas_gp1, 
        'y_preds_gp2' : y_preds_gp2,
        'y_sigmas_gp2' : y_sigmas_gp2,
        'y_preds_gp3' : y_preds_gp3,
        'y_sigmas_gp3' : y_sigmas_gp3,
        'y_preds_gp_h': y_preds_gp_h,
        'y_sigmas_gp_h': y_sigmas_gp_h,
        'pears_gp1' : pears_gp1_robust,
        'kts_gp1' : kts_gp1_robust,
        'pears_gp2' : pears_gp2_robust,
        'kts_gp2' : kts_gp2_robust,
        'pears_gp3' : pears_gp3_robust,
        'kts_gp3' : kts_gp3_robust,
        'pears_gp_h': pears_gp_h,
        'kts_gp_h': kts_gp_h
    }

    with open(f'{folder}/regression_data_{EXPERIMENT_NUMBER}.pkl', 'wb') as f:
        dill.dump(regression_data, f)
