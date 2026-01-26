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


def create_path_name(base: str, exp: str, refine: str, init_samples: int, starting: str = '', kernel_choice: str = ''): 
    d_path = f'{base}/{exp}_{refine}_{init_samples}'
    if starting != '':
        d_path = f'{d_path}/{starting}'
    p = Path(d_path)
    return p, d_path


def pickle_data(data, name: str, refine: str, exp: str, init_samples: int, base: str = "results", starting: str ='', kernel_choice: str = ''):
    p, d_path = create_path_name(exp=exp, refine=refine, base=base, init_samples=init_samples, starting=starting, kernel_choice=kernel_choice)
    p.mkdir(parents=True, exist_ok=True)
    if kernel_choice != '':
        f_name = f'{name}_{kernel_choice}'
    else:
        f_name = f'{name}'
    with open(f'{d_path}/{f_name}.pkl', 'wb') as f: 
        dill.dump(data, f)


starting = datetime.now().strftime("%Y%m%d_%H%M%S")
refine = 'ref'
exp = 'ode_1_bo'
kernel_choice = "WL"  # alternatively:  hWL
init_sample_size: int = 50 # 10, 50
budget = (10, 10, 10) # TODO: measure time for whole BO process and increase or decrease budget accordingly, to run within 24hrs

repo = ODE_1_Repository(linear_feature_dimensions=[1, 2, 3, 4], constant_values=[0, 1, -1], learning_rate_values=[1e-2, 5e-3 ,1e-3],
                        n_epoch_values=[1000])

edge = (("swap", 0, 1), 1, 1)


def parallel_edges(n):
    if n <= 0:
        raise ValueError("n must be positive")
    else:
        return (("swap", 0, n), n, n)

# Load pre generated data for the training
data = torch.load('data/ode1_dataset.pth')
x_data_fucking_side_effects = data['x_train']
y_data_fucking_side_effects = data['y_train']
x_test_data_fucking_side_effects = data['x_test']
y_test_data_fucking_side_effects = data['y_test']


def f_obj(x, y, x_test, y_test):
    #learner = t.interpret(repo.pytorch_function_algebra())
    return lambda t: t.interpret(repo.pytorch_function_algebra())(x, y, x_test, y_test)

# target that synthesizes exactly the one solution, from which the data was generated
target_solution = Constructor("Learner", Constructor("DAG",
                                                     Constructor("input", Literal(1))
                                                     & Constructor("output", Literal(1))
                                                     & Constructor("structure", Literal(
                                                         (
                                                            (
                                                                (repo.Copy(2), 1, 2),
                                                            ),
                                                            (
                                                                (repo.Linear(1, 1, True), 1, 1),
                                                                (repo.Linear(1, 1, True), 1, 1),
                                                            ),
                                                            (
                                                                edge, 
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

target = target_solution


synthesizer = SearchSpaceSynthesizer(repo.specification(), {})
search_space = synthesizer.construct_search_space(target_solution).prune()
test = search_space.enumerate_trees(target, 10)
test_list = list(test)
print(f"Number of trees found: {len(test_list)}") #  should be 1, otherwise target_solution is wrong
data_generating_tree = test_list[0]

# pickle the data generating tree, to know the optimal structure
pickle_data(data_generating_tree, name='data_generating_tree', refine=refine, exp=exp, starting=starting, init_samples=init_sample_size, kernel_choice=kernel_choice)


# derived target for the actual ODE1 dataset/best structure
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
                                                                  #None,  # left_out, 1-gate, right
                                                                  #None,  # left_out, right_out
                                                                  #None
                                                              )
                                                          )))
                                   & Constructor("Loss", Constructor("type", Literal(None)))
                                   & Constructor("Optimizer", Constructor("type", Literal(None)))
                                   & Constructor("epochs", Literal(1000))
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
    target = target_from_trapezoid1

    synthesizer = SearchSpaceSynthesizer(repo.specification(), {})

    search_space = synthesizer.construct_search_space(target).prune()
    print("finished synthesis")
    # Andreas, uncomment this to check that the search space isn't empty and if the target is ok, comment it out afterwards

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
                              generation_limit=50, elitism=1,
                              enforce_diversity=False)

    """
    test = search_space.enumerate_trees(target, 10)

    test_list = list(test)
    print(f"Number of trees found: {len(test_list)}")
    """

    pickle_data(search_space, name='search_space_1', refine=refine, exp=exp, starting=starting, init_samples=init_sample_size, kernel_choice=kernel_choice)

    _, d_path = create_path_name(exp=exp, refine='', base='data', init_samples=init_sample_size)
    d_path = f'{d_path}/starting_points.pkl'
    p = Path(d_path)
    if p.exists():
        print(f'Existing data: {d_path}')
        with open(d_path, 'rb') as f:
            existing_data = dill.load(f)
        x_gp = existing_data['x_gp']
        y_gp = existing_data['y_gp']
    else:
        print(f'Data does not exist')
        # terms = search_space.sample(init_sample_size, target)

        next = search_space.sample_tree(target)
        terms = []
        while len(terms) < init_sample_size:
            print(len(terms))
            is_duplicate = False
            for tree in terms:
                k = kernel._f(next, tree)  # kernel1 should be enough
                if k > 0.99:  # almost identical
                    is_duplicate = True
                    break
            if not is_duplicate:
                terms.append(next)
            next = search_space.sample_tree(target)

        x_gp = list(terms)
        y_gp = [f_obj(x_data_fucking_side_effects, x_data_fucking_side_effects, x_data_fucking_side_effects, x_data_fucking_side_effects)(t) for t in x_gp]

        tmp = []
        for term in x_gp:
            pickle = term.interpret(repo.pickle_algebra())
            tmp.append(pickle)
        x_gp = tmp

        # Safe the "starting points" for BO and load them, instead of resampling every time
        starting_points = {
            'x_gp': x_gp, 
            'y_gp': y_gp
        }
        pickle_data(starting_points, name='starting_points', refine='', exp=exp, base='data', starting='', init_samples=init_sample_size)

    # Unpickle the starting points like it is done for loading to keep equally between runs
    tmp = []
    print("Unpickling existing data")
    for idx, x_pickle in enumerate(x_gp):
        # print(f'Idx: {idx}')
        target_ = repo.from_pickle(x_pickle)
        search_space_tmp = synthesizer.construct_search_space(target_)
        terms = list(search_space_tmp.enumerate_trees(target_, 10))
        # print(f'Num Terms: {len(list(terms))}')
        assert(len(list(terms)) == 1)
        tmp.append(list(terms)[0])
    x_gp = tmp

    print("duplicates in data:")
    print("X: ", len(x_gp) - len(set(x_gp)))
    print("Y: ", len(y_gp) - len(set(y_gp)))

    print("X should not have any duplicates!")
    print("If Y has duplicates, either the objective function is not injective or its a rounding error.")


    """
    TODO Measure the time and alter the parameters accordingly
    Generation Limit >= 10
    Population Size: >= 50
    """

    start = time.time()

    # result is a dictionary with keys: "best_tree", "x", "y", "gp_model"
    result = bo.bayesian_optimisation(n_iters=budget[0], obj_fun=f_obj(x_data_fucking_side_effects, x_data_fucking_side_effects, x_data_fucking_side_effects, x_data_fucking_side_effects),
                                      x0=x_gp, y0=y_gp, n_pre_samples=init_sample_size,
                                      greater_is_better=False, ei_xi=0.1)  # adjusting ei_xi allows to trade off exploration vs exploitation. small xi (0.001) -> exploitation, large xi (0.1)-> exploration
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
    result['elapsed_time'] = end - start
    # safe results (values from result, best_y, time etc.)
    pickle_data(result, name='result_1', refine=refine, exp=exp, starting=starting, init_samples=init_sample_size, kernel_choice=kernel_choice)
    pickle_data(kernel, name='kernel_1', refine=refine, exp=exp, starting=starting, init_samples=init_sample_size, kernel_choice=kernel_choice)

    ##############################################################

    next_target = result["best_tree"].interpret(repo.to_structure_2_algebra())
    print("Next Target: ", next_target)
    synthesizer = SearchSpaceSynthesizer(repo.specification(), {})

    search_space = synthesizer.construct_search_space(next_target).prune()
    print("finished synthesis")

    # safe next_target and its search space. Maybe measure synthesis time?
    pickle_data(search_space, name='search_space_2', refine=refine, exp=exp, starting=starting, init_samples=init_sample_size, kernel_choice=kernel_choice)
    pickle_data(next_target, name='next_target_2', refine=refine, exp=exp, starting=starting, init_samples=init_sample_size, kernel_choice=kernel_choice)

    if kernel_choice == "WL":
        kernel = WeisfeilerLehmanKernel(n_iter=1, to_grakel_graph=to_grakel_graph_2)
    elif kernel_choice == "hWL":
        kernel = result["gp_model"].kernel_   # kernel with optimized hyperparameters from previous BO run
    else:
        raise ValueError(f"Unknown kernel choice: {kernel_choice}")

    bo = BayesianOptimization(search_space, next_target, kernel=kernel,
                              kernel_optimizer=kernel.optimize_hyperparameter, n_restarts_optimizer=2,
                              population_size=100, tournament_size=5,
                              crossover_rate=0.85, mutation_rate=0.35,
                              generation_limit=50, elitism=1,
                              enforce_diversity=False)

    start = time.time()

    # result is a dictionary with keys: "best_tree", "x", "y", "gp_model"
    result = bo.bayesian_optimisation(n_iters=budget[1], obj_fun=f_obj(x_data_fucking_side_effects, x_data_fucking_side_effects, x_data_fucking_side_effects, x_data_fucking_side_effects),
                                      x0=[result["best_tree"]], y0=[best_y], n_pre_samples=init_sample_size,
                                      greater_is_better=False,
                                      ei_xi=0.01)  # adjusting ei_xi allows to trade off exploration vs exploitation. small xi (0.001) -> exploitation, large xi (0.1)-> exploration
    end = time.time()
    print("Best tree found:")
    print(result["best_tree"].interpret(repo.pretty_term_algebra()))
    print("The following data was generated:")
    for x, y in zip(result["x"], result["y"]):
        print(f"Tree: {x.interpret(repo.pretty_term_algebra())}, Test Loss: {y}")
        if x == result["best_tree"]:
            best_y = y
    print(f'Elapsed Time: {end - start}')
    # safe results (values from result, best_y, time etc.)
    result['elapsed_time'] = end - start 
    pickle_data(result, name='result_2', refine=refine, exp=exp, starting=starting, init_samples=init_sample_size, kernel_choice=kernel_choice)
    pickle_data(kernel, name='kernel_2', refine=refine, exp=exp, starting=starting, init_samples=init_sample_size, kernel_choice=kernel_choice)

    ##############################################################
    last_target = result["best_tree"].interpret(repo.to_structure_2_algebra())

    print("Last Target: ", last_target)
    synthesizer = SearchSpaceSynthesizer(repo.specification(), {})

    search_space = synthesizer.construct_search_space(last_target).prune()
    print("finished synthesis")
    # safe last_target and its search space. Maybe measure synthesis time?
    pickle_data(search_space, name='search_space_3', refine=refine, exp=exp, starting=starting, init_samples=init_sample_size, kernel_choice=kernel_choice)
    pickle_data(next_target, name='next_target_3', refine=refine, exp=exp, starting=starting, init_samples=init_sample_size, kernel_choice=kernel_choice)
    if kernel_choice == "WL":
        kernel = WeisfeilerLehmanKernel(n_iter=1, to_grakel_graph=to_grakel_graph_3)
    elif kernel_choice == "hWL":
        kernel = result["gp_model"].kernel_  # kernel with optimized hyperparameters from previous BO run
    else:
        raise ValueError(f"Unknown kernel choice: {kernel_choice}")

    bo = BayesianOptimization(search_space, last_target, kernel=kernel,
                              kernel_optimizer=kernel.optimize_hyperparameter, n_restarts_optimizer=2,
                              population_size=100, tournament_size=5,
                              crossover_rate=0.85, mutation_rate=0.35,
                              generation_limit=50, elitism=1,
                              enforce_diversity=False)

    start = time.time()

    # result is a dictionary with keys: "best_tree", "x", "y", "gp_model"
    result = bo.bayesian_optimisation(n_iters=budget[2], obj_fun=f_obj(x_data_fucking_side_effects, x_data_fucking_side_effects, x_data_fucking_side_effects, x_data_fucking_side_effects), x0=[result["best_tree"]], y0=[best_y],
                                      n_pre_samples=init_sample_size,
                                      greater_is_better=False,
                                      ei_xi=0.001)  # adjusting ei_xi allows to trade off exploration vs exploitation. small xi (0.001) -> exploitation, large xi (0.1)-> exploration
    end = time.time()
    print("Best tree found:")
    print(result["best_tree"].interpret(repo.pretty_term_algebra()))
    print("The following data was generated:")
    for x, y in zip(result["x"], result["y"]):
        print(f"Tree: {x.interpret(repo.pretty_term_algebra())}, Test Loss: {y}")
        if x == result["best_tree"]:
            best_y = y
    print(f'Elapsed Time: {end - start}')
    # safe results (values from result, best_y, time etc.)
    result['elapsed_time'] = end - start
    pickle_data(result, name='result_3', refine=refine, exp=exp, starting=starting, init_samples=init_sample_size, kernel_choice=kernel_choice)
    pickle_data(kernel, name='kernel_3', refine=refine, exp=exp, starting=starting, init_samples=init_sample_size, kernel_choice=kernel_choice)

    # compare result["best_tree"] to data generating tree, if available
    # comparison can be done via kernels, to measure how similar the structures are
