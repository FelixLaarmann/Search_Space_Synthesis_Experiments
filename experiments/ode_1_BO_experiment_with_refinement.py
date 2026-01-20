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

repo = ODE_1_Repository(linear_feature_dimensions=[1, 2, 3], constant_values=[0, 1, -1], learning_rate_values=[1e-2],
                        n_epoch_values=[10000], dimensions=[1,2,3,4])

edge = (("swap", 0, 1), 1, 1)


def parallel_edges(n):
    if n <= 0:
        raise ValueError("n must be positive")
    else:
        return (("swap", 0, n), n, n)

# Load pre generated data for the training
data = torch.load('../data/TrapezoidNet.pth') # TODO: dataset for actual ODE1 target, since trapezoid would be ODE2
x = data['x_train']
y = data['y_train']
x_test = data['x_test']
y_test = data['y_test']


def f_obj(t):
    learner = t.interpret(repo.pytorch_function_algebra())
    return learner(x, y, x_test, y_test)

# TODO: target that synthesizes exactly the one solution, from which the data was generated
"""
target_solution = None

target = target_solution

synthesizer = SearchSpaceSynthesizer(repo.specification(), {})
search_space = synthesizer.construct_search_space(target_solution).prune()
test = search_space.enumerate_trees(target, 10)
test_list = list(test)
print(f"Number of trees found: {len(test_list)}") #  should be 1, otherwise target_solution is wrong
data_generating_tree = test_list[0]
"""
# TODO: pickle the data generating tree, to know the optimal structure

# TODO: derived target for the actual ODE1 dataset/best structure
target_from_trapezoid1 = Constructor("Learner", Constructor("DAG",
                                                          Constructor("input", Literal(1))
                                                          & Constructor("output", Literal(1))
                                                          & Constructor("structure", Literal(
                                                              (
                                                                  None,
                                                                  None,  # left, split, right
                                                                  None,  # left, gate, right
                                                                  None,  # left, gate, right
                                                                  None,  # left_out, -gate, right
                                                                  None,  # left_out, 1-gate, right
                                                                  None,  # left_out, right_out
                                                                  None
                                                              )
                                                          )))
                                   & Constructor("Loss", Constructor("type", Literal(None)))
                                   & Constructor("Optimizer", Constructor("type", Literal(None)))
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

    init_sample_size = 10
    budget = (10, 10, 10) # TODO: measure time for whole BO process and increase or decrease budget accordingly, to run within 24hrs
    kernel_choice = "hWL"  # alternatively: "WL"

    target = target_from_trapezoid1

    synthesizer = SearchSpaceSynthesizer(repo.specification(), {})

    search_space = synthesizer.construct_search_space(target).prune()
    print("finished synthesis")
    # TODO: Andreas, uncomment this to check that the search space isn't empty and if the target is ok, comment it out afterwards
    """
    test = search_space.enumerate_trees(target, 10)

    test_list = list(test)
    print(f"Number of trees found: {len(test_list)}")
    """
    # TODO: if the search space looks good, pickle it

    terms = search_space.sample(init_sample_size, target)
    x_gp = list(terms)
    y_gp = [f_obj(t) for t in x_gp]

    # TODO: Safe the "starting points" for BO and load them, instead of resampling every time

    print("duplicates in data:")
    print("X: ", len(x_gp) - len(set(x_gp)))
    print("Y: ", len(y_gp) - len(set(y_gp)))

    print("X should not have any duplicates!")
    print("If Y has duplicates, either the objective function is not injective or its a rounding error.")

    if kernel_choice == "WL":
        kernel = WeisfeilerLehmanKernel(n_iter=1, to_grakel_graph=to_grakel_graph_1)
    elif kernel_choice == "hWL":
        kernel = OptimizableHierarchicalWeisfeilerLehmanKernel(to_grakel_graph1=to_grakel_graph_1,
                                                            to_grakel_graph2=to_grakel_graph_2,
                                                            to_grakel_graph3=to_grakel_graph_3,
                                                            weight1=0.4, weight2=0.3, weight3=0.3,
                                                            n_iter1=1, n_iter2=1, n_iter3=1)
    else:
        raise ValueError(f"Unknown kernel choice: {kernel_choice}")

    bo = BayesianOptimization(search_space, target, kernel=kernel,
                              kernel_optimizer=kernel.optimize_hyperparameter, n_restarts_optimizer=2,
                              population_size=100, tournament_size=5,
                              crossover_rate=0.85, mutation_rate=0.35,
                              generation_limit=30, elitism=1,
                              enforce_diversity=False)

    start = time.time()

    # result is a dictionary with keys: "best_tree", "x", "y", "gp_model"
    result = bo.bayesian_optimisation(n_iters=budget[0], obj_fun=f_obj, x0=x_gp, y0=y_gp, n_pre_samples=init_sample_size,
                                      greater_is_better=False, ei_xi=0.01)  # adjusting ei_xi allows to trade off exploration vs exploitation
    end = time.time()
    print("Best tree found:")
    print(result["best_tree"].interpret(repo.pretty_term_algebra()))
    print("The following data was generated:")
    best_y = None
    for x, y in zip(result["x"], result["y"]):
        print(f"Tree: {x.interpret(repo.pretty_term_algebra())}, Test Loss: {y}")
        if x == result["best_tree"]:
            best_y = y
    print(f'Elapsed Time: {end - start}')
    next_target = result["best_tree"].interpret(repo.to_structure_2_algebra())
    print("Next Target: ", next_target)
    synthesizer = SearchSpaceSynthesizer(repo.specification(), {})

    search_space = synthesizer.construct_search_space(next_target).prune()
    print("finished synthesis")
    if kernel_choice == "WL":
        kernel = WeisfeilerLehmanKernel(n_iter=1, to_grakel_graph=to_grakel_graph_2)
    elif kernel_choice == "hWL":
        kernel = result["gp_model"].kernel   # kernel with optimized hyperparameters from previous BO run
    else:
        raise ValueError(f"Unknown kernel choice: {kernel_choice}")

    bo = BayesianOptimization(search_space, next_target, kernel=kernel,
                              kernel_optimizer=kernel.optimize_hyperparameter, n_restarts_optimizer=2,
                              population_size=100, tournament_size=5,
                              crossover_rate=0.85, mutation_rate=0.35,
                              generation_limit=30, elitism=1,
                              enforce_diversity=False)

    start = time.time()

    # result is a dictionary with keys: "best_tree", "x", "y", "gp_model"
    result = bo.bayesian_optimisation(n_iters=budget[1], obj_fun=f_obj, x0=[result["best_tree"]], y0=best_y, n_pre_samples=init_sample_size,
                                      greater_is_better=False,
                                      ei_xi=0.01)  # adjusting ei_xi allows to trade off exploration vs exploitation
    end = time.time()
    print("Best tree found:")
    print(result["best_tree"].interpret(repo.pretty_term_algebra()))
    print("The following data was generated:")
    for x, y in zip(result["x"], result["y"]):
        print(f"Tree: {x.interpret(repo.pretty_term_algebra())}, Test Loss: {y}")
    print(f'Elapsed Time: {end - start}')
    last_target = result["best_tree"].interpret(repo.to_structure_2_algebra())


    # TODO: compare result["best_tree"] to data generating tree, if available with the kernels