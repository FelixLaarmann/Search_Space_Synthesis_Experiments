from cl3s import (SpecificationBuilder, Constructor, Literal, Var, Group, DataGroup,
                  DerivationTree, SearchSpaceSynthesizer, BayesianOptimization, WeisfeilerLehmanKernel)

import torch
import torch.nn as nn
import torch.optim as optim

from synthesis.utils import generate_data

import re

from grakel.utils import graph_from_networkx

import networkx as nx

from synthesis.old_ode_repo import ODE_Trapezoid_Repository

repo_lte = ODE_Trapezoid_Repository(dimensions=[1, 2, 3, 4], linear_feature_dimensions=[1,], sharpness_values=[2],
                                    threshold_values=[0], constant_values=[0, 1, -1], learning_rate_values=[1e-2],
                                    n_epoch_values=[10000])

edge = (("swap", 0, 1), 1, 1)


def parallel_edges(n):
    if n <= 0:
        raise ValueError("n must be positive")
    else:
        return (("swap", 0, n), n, n)

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

target_from_trapezoid1 = Constructor("Learner", Constructor("DAG",
                                                            Constructor("input", Literal(None))
                                                            & Constructor("output", Literal(1))
                                                            & Constructor("structure", Literal(
                                                                (
                                                                    None,  # left, split, right
                                                                    None,  # left, gate, right
                                                                    None,  # left_out, -gate, right
                                                                    None,  # left_out, 1-gate, right
                                                                    None,  # left_out, right_out
                                                                    None
                                                                )
                                                            )))
                                     & Constructor("Loss", Constructor("type", Literal(repo_lte.MSEloss())))
                                     & Constructor("Optimizer", Constructor("type", Literal(repo_lte.Adam(1e-2))))
                                     & Constructor("epochs", Literal(10000))
                                     )

target_from_trapezoid2 = Constructor("Learner", Constructor("DAG",
                                                            Constructor("input", Literal(None))
                                                            & Constructor("output", Literal(1))
                                                            & Constructor("structure", Literal(
                                                                (
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
                                     & Constructor("Loss", Constructor("type", Literal(repo_lte.MSEloss())))
                                     & Constructor("Optimizer", Constructor("type", Literal(repo_lte.Adam(1e-2))))
                                     & Constructor("epochs", Literal(10000))
                                     )

target_from_trapezoid3 = Constructor("Learner", Constructor("DAG",
                                                            Constructor("input", Literal(3))
                                                            & Constructor("output", Literal(1))
                                                            & Constructor("structure", Literal(
                                                                (
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
                                     & Constructor("Loss", Constructor("type", Literal(repo_lte.MSEloss())))
                                     & Constructor("Optimizer", Constructor("type", Literal(repo_lte.Adam(1e-2))))
                                     & Constructor("epochs", Literal(10000))
                                     )

def to_grakel_graph(repo, t):
    edgelist = t.interpret(repo.edgelist_algebra())

    G = nx.MultiDiGraph()
    G.add_edges_from(edgelist)

    relabel = {n: re.sub("[)][(][-]*[0-9]*[.][0-9]*[,]\s[-]*[0-9]*[.][0-9]*[)]", ")", n)
               for n in G.nodes()}

    for n in G.nodes():
        G.nodes[n]['symbol'] = relabel[n]

    gk_graph = graph_from_networkx([G.to_undirected()], node_labels_tag='symbol')

    return gk_graph

def to_grakel_graph_0(repo, t):
    edgelist = t.interpret(repo.edgelist_algebra())

    G = nx.MultiDiGraph()
    G.add_edges_from(edgelist)

    relabel = {n: "Node"
               for n in G.nodes()}

    for n in G.nodes():
        G.nodes[n]['symbol'] = relabel[n]

    gk_graph = graph_from_networkx([G.to_undirected()], node_labels_tag='symbol')

    return gk_graph

def to_grakel_graph_1(repo, t):
    edgelist = t.interpret(repo.edgelist_algebra())

    G = nx.MultiDiGraph()
    G.add_edges_from(edgelist)

    relabel = {n: "Linear" if "Linear" in n else "Sigmoid" if "Sigmoid" in n else "ReLu" if "ReLu" in n else
                    "Sharpness_Sigmoid" if "Sharpness_Sigmoid" in n else "LTE" if "LTE" in n else
                    "Sum" if "Sum" in n else
                    "Product" if "Product" in n else "Node"
               for n in G.nodes()}

    for n in G.nodes():
        G.nodes[n]['symbol'] = relabel[n]

    gk_graph = graph_from_networkx([G.to_undirected()], node_labels_tag='symbol')

    return gk_graph

if __name__ == "__main__":

    inputs = {"repo": "ODE_LTE", "init_sample_size": 10, "budget": 10}

    if inputs["repo"] == "ODE_LTE":
        target = target_from_trapezoid3
        repo = repo_lte
    else: # add other repos here
        target = target_from_trapezoid1
        repo = repo_lte

    synthesizer = SearchSpaceSynthesizer(repo.specification(), {})

    search_space = synthesizer.construct_search_space(target).prune()
    print("finished synthesis")

    kernel = WeisfeilerLehmanKernel(n_iter=1, to_grakel_graph=lambda t: to_grakel_graph(repo, t))

    bo = BayesianOptimization(search_space, target, kernel=kernel, population_size=10, tournament_size=3,
                              crossover_rate=0.85, mutation_rate=0.4, generation_limit=20, elitism=1,
                              enforce_diversity=False)

    # data generation
    # TODO: Andreas

    class TrapezoidNetPure(nn.Module):
        def __init__(self, random_weights=False, sharpness=None):
            super().__init__()

            self.split = nn.Linear(1, 1, bias=True)
            self.left = nn.Linear(1, 1, bias=True)
            self.right = nn.Linear(1, 1, bias=True)
            self.sharpness = sharpness

            if not random_weights:
                with torch.no_grad():
                    # For left branch (x <= 0): we want output = x + 10
                    # So left(x) = x + 10 => weight = 1, bias = 10
                    self.left.weight.data.fill_(1.0)
                    self.left.bias.data.fill_(10.0)

                    # For right branch (x > 0): we want output = 10 - x
                    # So right(x) = 10 - x => weight = -1, bias = 10
                    self.right.weight.data.fill_(-1.0)
                    self.right.bias.data.fill_(10.0)

                self.split.weight.data.fill_(1.0)
                self.split.bias.data.fill_(0.0)

        def forward(self, x):
            if not self.sharpness:
                gate = (self.split(x) <= 0).float()
            else:
                gate = torch.sigmoid(-self.sharpness * self.split(x))

            left_out = self.left(x) * gate
            right_out = self.right(x) * (1 - gate)

            return left_out + right_out


    true_model = TrapezoidNetPure()

    # Generate data is a bit noisy to make it closer to the real world
    # test data is a little bit out of distribution because we change xmin/xmax a little bit
    x, y = generate_data(true_model, xmin=-10, xmax=10, n_samples=1_000, eps=1e-4)
    x_test, y_test = generate_data(true_model, xmin=-15, xmax=15, n_samples=1_000, eps=1e-4)

    def f_obj(t):
        learner = t.interpret(repo.pytorch_function_algebra())
        return learner(x, y, x_test, y_test)

    best_tree, X, Y = bo.bayesian_optimisation(inputs["init_sample_size"], f_obj,
                                               n_pre_samples=inputs["init_sample_size"], greater_is_better=False) # minimize f_obj

    print("Best tree found:")
    print(best_tree.interpret(repo.pretty_term_algebra()))
    print("The following data was generated:")
    for x, y in zip(X, Y):
        print(f"Tree: {x.interpret(repo.pretty_term_algebra())}, Test Loss: {y}")