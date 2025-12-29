from synthesis.labeled_dag import Labeled_DAG_Repository
from cl3s import (Constructor, Literal, SearchSpaceSynthesizer)
from cl3s import WeisfeilerLehmanKernel
from cl3s import ExpectedImprovement

from grakel.utils import graph_from_networkx

import networkx as nx
import matplotlib.pyplot as plt
from itertools import accumulate

import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor

def plot_term(t):
    f, inputs = t.interpret(repo.edgelist_algebra())
    edgelist, to_outputs, pos_A = f((-3.8, -3.8), ["input" for _ in range(0, inputs)])
    edgelist = edgelist + [(o, "output") for o in to_outputs]

    pos_A = pos_A | {"input": (-5.5, -3.8), "output": (max([x for x, y in pos_A.values()]) + 2.5, -3.8)}

    G = nx.MultiDiGraph()
    G.add_edges_from(edgelist)

    relabel = {n: ("A" if "A" in n
                   else "B" if "B" in n
                   else "C" if "C" in n
                   else n)
               for n in G.nodes()}

    for n in G.nodes():
        G.nodes[n]['symbol'] = relabel[n]

    connectionstyle = [f"arc3,rad={r}" for r in accumulate([0.3] * 4)]

    plt.figure(figsize=(25, 25))

    node_size = 3000
    nx.draw_networkx_nodes(G, pos_A, node_size=node_size, node_color='lightblue', alpha=0.5, margins=0.05)
    nx.draw_networkx_labels(G, pos_A, labels=relabel, font_size=6, font_weight="bold")
    nx.draw_networkx_edges(G, pos_A, edge_color="black", connectionstyle=connectionstyle, node_size=node_size,
                           width=2)
    plt.figtext(0.01, 0.02, t.interpret(repo.pretty_term_algebra()), fontsize=14)

    plt.show()

def to_grakel_graph(t):
    f, inputs = t.interpret(repo.edgelist_algebra())
    edgelist, to_outputs, pos_A = f((-3.8, -3.8), ["input" for _ in range(0, inputs)])
    edgelist = edgelist + [(o, "output") for o in to_outputs]

    G = nx.MultiDiGraph()
    G.add_edges_from(edgelist)

    relabel = {n: ("A" if "A" in n
                   else "B" if "B" in n
                   else "C" if "C" in n
                   else n)
               for n in G.nodes()}

    for n in G.nodes():
        G.nodes[n]['symbol'] = relabel[n]

    gk_graph = graph_from_networkx([G.to_undirected()], node_labels_tag='symbol')

    return gk_graph


if __name__ == "__main__":
    repo = Labeled_DAG_Repository(labels=["A", "B", "C"], dimensions=range(1, 4))

    # This target is isomorphic to a 4-tuple of the labels.
    # Therefore, we the search space contains 3^4 = 81 elements.
    target_linear = Constructor("DAG",
                                Constructor("input", Literal(1))
                                & Constructor("output", Literal(1))
                                & Constructor("structure", Literal(
                                    ((None,), (None,), (None,), (None,))
                                )))

    # 135 terms, because we may have a parallel edge in the middle layer.
    target_parallel = Constructor("DAG",
                                Constructor("input", Literal(1))
                                & Constructor("output", Literal(1))
                                & Constructor("structure", Literal(
                                    ((None,), (None, None), (None,))
                                )))

    target_bigger = Constructor("DAG",
                                  Constructor("input", Literal(1))
                                  & Constructor("output", Literal(1))
                                  & Constructor("structure", Literal(
                                      ((None,), (None, None), (None, None), (None,))
                                  )))

    target = target_bigger

    synthesizer = SearchSpaceSynthesizer(repo.specification(), {})

    search_space = synthesizer.construct_search_space(target).prune()
    print("finish synthesis, start enumerate")

    terms = search_space.enumerate_trees(target, 3000)

    print("enumeration finished")

    #term = terms.__next__()

    terms_list = list(terms)

    term = terms_list[0]

    print(target)

    print(f"number of terms: {len(terms_list)}")
    """
    for t in terms_list:
        print(t.interpret(repo.pretty_term_algebra()))
    """

    kernel = WeisfeilerLehmanKernel(to_grakel_graph=to_grakel_graph)

    # Plot the kernel matrix for the first 10 terms
    L_small = [t.interpret(repo.pretty_term_algebra()) for t in terms_list[:10]]

    X_small = np.array(terms_list[:10])
    K = kernel(X_small)
    D = kernel.diag(X_small)

    plt.figure(figsize=(8, 5))
    plt.imshow(np.diag(D ** -0.5).dot(K).dot(np.diag(D ** -0.5)))
    plt.xticks(np.arange(len(X_small)), L_small)
    plt.yticks(np.arange(len(X_small)), L_small)
    plt.title("Term similarity under the kernel")
    plt.show()

    def f_obj(t):
        return kernel._f(term, t)

    print(f"Our objective function will use the kernel to compare terms to the term:\n{term.interpret(repo.pretty_term_algebra())}")

    X = np.array(terms_list)
    y = np.array([f_obj(t) for t in terms_list])

    L = np.array([t.interpret(repo.pretty_term_algebra()) for t in terms_list])

    K = kernel(X)
    D = kernel.diag(X)

    plt.figure(figsize=(8, 5))
    plt.imshow(np.diag(D ** -0.5).dot(K).dot(np.diag(D ** -0.5)))
    plt.xticks(np.arange(len(X)), L)
    plt.yticks(np.arange(len(X)), L)
    plt.title("Term similarity under the kernel")
    plt.show()

    plt.plot(L, y, linestyle="dotted")
    plt.xlabel("$t$")
    plt.ylabel("$f obj(t)$")
    _ = plt.title("True generative process")
    plt.show()

    rng = np.random.RandomState(1)
    training_indices = rng.choice(np.arange(y.size), size=10, replace=False)
    X_train, y_train = X[training_indices], y[training_indices]

    L_train = np.array([t.interpret(repo.pretty_term_algebra()) for t in X_train])

    gaussian_process = GaussianProcessRegressor(kernel=kernel, optimizer=None, normalize_y=False)
    gaussian_process.fit(X_train, y_train)

    mean_prediction, std_prediction = gaussian_process.predict(X, return_std=True)

    ei = ExpectedImprovement(gaussian_process, True)

    y_ei = np.array([ei(t) for t in X])

    plt.plot(L, y, label="f obj", linestyle="dotted")
    plt.scatter(L_train, y_train, label="Observations")
    plt.plot(L, mean_prediction, label="Mean prediction")
    plt.plot(L, y_ei, label="EI", linestyle="dashdot")
    plt.fill_between(
        L.ravel(),
        mean_prediction - 1.96 * std_prediction,
        mean_prediction + 1.96 * std_prediction,
        alpha=0.5,
        label=r"95% confidence interval",
    )
    plt.legend()
    plt.xlabel("$t$")
    plt.ylabel("$f obj(t)$")
    _ = plt.title("Gaussian process regression on noise-free dataset")
    plt.show()
