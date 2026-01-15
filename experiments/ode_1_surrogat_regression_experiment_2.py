from cl3s import (SpecificationBuilder, Constructor, Literal, Var, Group, DataGroup,
                  DerivationTree, SearchSpaceSynthesizer, BayesianOptimization, WeisfeilerLehmanKernel)
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





repo = ODE_1_Repository(linear_feature_dimensions=[1, 2, 3], constant_values=[0, 1, -1], learning_rate_values=[1e-2],
                        n_epoch_values=[10000], dimensions=[1,2,3,4])

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
target_from_trapezoid1 = Constructor("Learner", Constructor("DAG",
                                                            Constructor("input", Literal(1))
                                                            & Constructor("output", Literal(1))
                                                            & Constructor("structure", Literal(
                                                                (
                                                                    None,
                                                                    None,  # left, split, right
                                                                    None,  # left, gate, right
                                                                    None,  # left_out, -gate, right
                                                                    None,  # left_out, 1-gate, right
                                                                    None,  # left_out, right_out
                                                                    None
                                                                )
                                                            )))
                                     & Constructor("Loss", Constructor("type", Literal(None)))
                                     & Constructor("Optimizer", Constructor("type", Literal(repo.Adam(1e-2))))
                                     & Constructor("epochs", Literal(10000))
                                     )

target_from_trapezoid2 = Constructor("Learner", Constructor("DAG",
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
                                                                        None
                                                                    ),  # left, split, right
                                                                    (
                                                                        None,
                                                                        None,
                                                                        None
                                                                    ),  # left, gate, right
                                                                    (
                                                                        None,
                                                                        None,
                                                                        None
                                                                    ),  # left_out, -gate, right
                                                                    (
                                                                        None,
                                                                        None,
                                                                        None
                                                                    ),  # left_out, 1-gate, right
                                                                    (
                                                                        None,
                                                                        None
                                                                    ),  # left_out, right_out
                                                                    (
                                                                        None,
                                                                    )
                                                                )
                                                            )))
                                     & Constructor("Loss", Constructor("type", Literal(None)))
                                     & Constructor("Optimizer", Constructor("type", Literal(repo.Adam(1e-2))))
                                     & Constructor("epochs", Literal(10000))
                                     )

target_from_trapezoid3 = Constructor("Learner", Constructor("DAG",
                                                                Constructor("input", Literal(1))
                                                                & Constructor("output", Literal(1))
                                                                & Constructor("structure", Literal(
                                                                    (
                                                                        (
                                                                            (None, 1, 3),
                                                                        ), # x -> (x,x,x)
                                                                        (
                                                                            (None, 1, 1),
                                                                            (None, 1, 1),
                                                                            (None, 1, 1)
                                                                        ),  # left, split, right
                                                                        (
                                                                            edge,
                                                                            (None, 1, 2),
                                                                            edge
                                                                        ),  # left, gate, right
                                                                        (
                                                                            (None, 2, 1),
                                                                            (None, 1, 1),
                                                                            edge
                                                                        ),  # left_out, -gate, right
                                                                        (
                                                                            edge,
                                                                            (None, 1, 1),
                                                                            edge
                                                                        ),  # left_out, 1-gate, right
                                                                        (
                                                                            edge,
                                                                            (None, 2, 1)
                                                                        ),  # left_out, right_out
                                                                        (
                                                                            (None, 2, 1),
                                                                        )
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

    relabel = {n: "Node"
               for n in G.nodes()}

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

    target = target_from_trapezoid2

    synthesizer = SearchSpaceSynthesizer(repo.specification(), {})

    search_space = synthesizer.construct_search_space(target).prune()
    print("finished synthesis")

    """
    test = search_space.enumerate_trees(target, 10)

    test_list = list(test)
    print(f"Number of trees found: {len(test_list)}")
    """

    kernel2 = WeisfeilerLehmanKernel(n_iter=1, to_grakel_graph=to_grakel_graph_2)

    kernel3 = WeisfeilerLehmanKernel(n_iter=1, to_grakel_graph=to_grakel_graph_3)

    gp2 = GaussianProcessRegressor(kernel=kernel2, optimizer=None, normalize_y=False)

    gp3 = GaussianProcessRegressor(kernel=kernel3, optimizer=None, normalize_y=False)

    # Load pre generated data for the training
    data = torch.load('../data/TrapezoidNet.pth')
    x = data['x_train']
    y = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']

    def f_obj(t):
        learner = t.interpret(repo.pytorch_function_algebra())
        return learner(x, y, x_test, y_test)

    terms = search_space.sample(sample_size, target)
    x_gp = list(terms)
    y_gp = [f_obj(t) for t in x_gp]

    print("duplicates in data:")
    print("X: ", len(x_gp) - len(set(x_gp)))
    print("Y: ", len(y_gp) - len(set(y_gp)))

    print("X should not have any duplicates!")
    print("If Y has duplicates, either the objective function is not injective or its a rounding error.")

    x_gp_test = np.array(x_gp[train_size:])
    y_gp_test = np.array(y_gp[train_size:])


    slice_size = int(train_size // plot_resolution)

    x_gp_train = [np.array(x_gp[i:i + slice_size]) for i in range(0, train_size, slice_size)] #[x_gp_train_1, x_gp_train_2, x_gp_train_3, x_gp_train_4, x_gp_train_5, x_gp_train_6, x_gp_train_7, x_gp_train_8]
    y_gp_train = [np.array(y_gp[i:i + slice_size]) for i in range(0, train_size, slice_size)] #[y_gp_train_1, y_gp_train_2, y_gp_train_3, y_gp_train_4, y_gp_train_5, y_gp_train_6, y_gp_train_7, y_gp_train_8]
    y_preds_gp2 = []
    y_sigmas_gp2 = []
    y_preds_gp3 = []
    y_sigmas_gp3 = []
    pears_gp2 = []
    kts_gp2 = []
    pears_gp3 = []
    kts_gp3 = []

    #print("data generated")

    for x_gp_i, y_gp_i in zip(x_gp_train, y_gp_train):
        K2 = kernel2(x_gp_i)
        D2 = kernel2.diag(x_gp_i)

        plt.figure(figsize=(8, 5))
        plt.imshow(np.diag(D2 ** -0.5).dot(K2).dot(np.diag(D2 ** -0.5)))
        plt.xticks(np.arange(len(x_gp_i)), range(1, len(x_gp_i) + 1))
        plt.yticks(np.arange(len(x_gp_i)), range(1, len(x_gp_i) + 1))
        plt.title("Term similarity under the kernel2")
        plt.show()

        gp2.fit(x_gp_i, y_gp_i)

        y_pred_next, sigma_next = gp2.predict(x_gp_test, return_std=True)

        y_preds_gp2.append(y_pred_next)
        y_sigmas_gp2.append(sigma_next)
        pears_gp2.append(pearsonr(y_gp_test, y_pred_next)[0])
        kts_gp2.append(kendalltau(y_gp_test, y_pred_next)[0])

        K3 = kernel3(x_gp_i)
        D3 = kernel3.diag(x_gp_i)

        plt.figure(figsize=(8, 5))
        plt.imshow(np.diag(D3 ** -0.5).dot(K3).dot(np.diag(D3 ** -0.5)))
        plt.xticks(np.arange(len(x_gp_i)), range(1, len(x_gp_i) + 1))
        plt.yticks(np.arange(len(x_gp_i)), range(1, len(x_gp_i) + 1))
        plt.title("Term similarity under the kernel3")
        plt.show()

        gp3.fit(x_gp_i, y_gp_i)

        y_pred_next, sigma_next = gp3.predict(x_gp_test, return_std=True)

        y_preds_gp3.append(y_pred_next)
        y_sigmas_gp3.append(sigma_next)
        pears_gp3.append(pearsonr(y_gp_test, y_pred_next)[0])
        kts_gp3.append(kendalltau(y_gp_test, y_pred_next)[0])

    plt.plot(range(train_size, (plot_resolution + 1)*train_size, train_size), kts_gp2, linestyle="dotted")
    plt.xlabel("# of samples")
    plt.ylabel("tau")
    _ = plt.title("Kendall Tau correlation for GP with kernel2")
    plt.show()

    plt.plot(range(train_size, (plot_resolution + 1)*train_size, train_size), pears_gp2, linestyle="dotted")
    plt.xlabel("# of samples")
    plt.ylabel("p")
    _ = plt.title("Pearson correlation for GP with kernel2")
    plt.show()

    plt.plot(range(train_size, (plot_resolution + 1) * train_size, train_size), kts_gp3, linestyle="dotted")
    plt.xlabel("# of samples")
    plt.ylabel("tau")
    _ = plt.title("Kendall Tau correlation for GP with kernel3")
    plt.show()

    plt.plot(range(train_size, (plot_resolution + 1) * train_size, train_size), pears_gp3, linestyle="dotted")
    plt.xlabel("# of samples")
    plt.ylabel("p")
    _ = plt.title("Pearson correlation for GP with kernel3")
    plt.show()