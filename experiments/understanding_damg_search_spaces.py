from synthesis.labeled_damg import Labeled_DAMG_Repository
from cosy import Constructor, Literal, Synthesizer

import networkx as nx
import matplotlib.pyplot as plt
from itertools import accumulate

"""
This file aims to explain how search spaces for directed acyclic multi-graphs (DAMGs) can be synthesized.
The explanation will be example-driven and a few well known neural network architectures will be given.

The most important part is to understand the request language.

If possible, we will use little ASCII-arts to illustrate the concepts.
A graph-art like

     B
  /     \
A          D - E
  \     /
     C

has to be read from left to right, since edges are directed.
Therefore the above ASCII-graph may be represented as  the following edge list:
[(A, B), (A, C), (B, D), (C, D), (D, E)]

So how do we represent such graphs in our request language?
First, we have to distinguish between two arts of composition:
Parallel composition and sequential composition.
Parallel composition means, that nodes B and C are in beside to each other, but there is no edge between them. 
We may also write B || C for the parallel composition of nodes B and C.
Sequential composition are essentially the edges. 
We may also write A ; B for the sequential composition of nodes A and B.
"""
# TODO: finish explanation/tutorial

target_linear = Constructor("DAG",
                            Constructor("input", Literal(1))
                            & Constructor("output", Literal(1))
                            & Constructor("structure", Literal(
                                ((None,), (None,), (None,), (None,))
                            )))

max_parallel_linear = 1

linear = (target_linear, max_parallel_linear)

edge = (("swap", 0, 1), 1, 1)

def parallel_edges(n):
    if n <= 0:
        raise ValueError("n must be positive")
    else:
        return (("swap", 0, n), n, n)

target_res_net_like = Constructor("DAG",
                            Constructor("input", Literal(1))
                            & Constructor("output", Literal(1))
                            & Constructor("structure", Literal(
                                ((None,), (None, edge), (None, edge), (None,))
                            )))

max_parallel_res_net_like = 2

res_net_like = (target_res_net_like, max_parallel_res_net_like)

target_u_net_like = Constructor("DAG",
                            Constructor("input", Literal(1))
                            & Constructor("output", Literal(1))
                            & Constructor("structure", Literal(
                                ((("Conv1D", 1, 2),), (edge, ("Maxpool1D", 1, 1),),
                                 (edge, ("Conv1D", 1, 2),), (edge, edge, ("Maxpool1D", 1, 1),),
                                 (edge, edge, ("Conv1D", 1, 1),), (edge, edge, ("Upsample", 1, 1),),
                                 (edge, ("Conv1D", 2, 1),), (edge, ("Upsample", 1, 1),),
                                 (("Conv1D", 2, 1),),
                                 (("Conv1D", 1, 1),),
                                 (("LinearLayer", 1, 1),),
                                 )
                            )))

target_u_net_like_nf = Constructor("DAG",
                            Constructor("input", Literal(1))
                            & Constructor("output", Literal(1))
                            & Constructor("structure", Literal(
                                ((("Conv1D", 1, 2),), (edge, ("Maxpool1D", 1, 1),),
                                 (edge, ("Conv1D", 1, 2),), (parallel_edges(2), ("Maxpool1D", 1, 1),),
                                 (parallel_edges(2), ("Conv1D", 1, 1),), (parallel_edges(2), ("Upsample", 1, 1),),
                                 (edge, ("Conv1D", 2, 1),), (edge, ("Upsample", 1, 1),),
                                 (("Conv1D", 2, 1),),
                                 (("Conv1D", 1, 1),),
                                 (("LinearLayer", 1, 1),),
                                 )
                            )))

max_parallel_u_net_like = 3

u_net_like = (target_u_net_like, max_parallel_u_net_like)

components = ["Conv1D", "LinearLayer", "Maxpool1D", "Dropout", "ReLU", "Sigmoid", "BatchNorm1D", "Upsample"]


#target, max_parallel = linear
target, max_parallel = res_net_like
#target, max_parallel = u_net_like


number_of_terms = 10

interpretation = "term" # "plotted_graph" # "term"

if __name__ == "__main__":
    repo = Labeled_DAMG_Repository(labels=components, dimensions=range(0, max_parallel + 1))

    print(target)

    synthesizer = Synthesizer(repo.specification(), {})

    search_space = synthesizer.construct_solution_space(target).prune()
    print("finish synthesis, start enumerate")
    terms = search_space.enumerate_trees(target, number_of_terms)

    for t in terms:
        if interpretation == "term":
            print(t.interpret(repo.pretty_term_algebra()))
        elif interpretation == "plotted_graph":
            f, inputs = t.interpret(repo.edgelist_algebra())
            edgelist, to_outputs, pos_A = f((-3.8, -3.8), ["input" for _ in range(0, inputs)])
            edgelist = edgelist + [(o, "output") for o in to_outputs]

            pos_A = pos_A | {"input": (-5.5, -3.8), "output": (max([x for x, y in pos_A.values()]) + 2.5, -3.8)}

            G = nx.MultiDiGraph()
            G.add_edges_from(edgelist)

            connectionstyle = [f"arc3,rad={r}" for r in accumulate([0.3] * 4)]

            plt.figure(figsize=(25, 25))

            pos_G = nx.bfs_layout(G, "input")
            node_size = 3000
            nx.draw_networkx_nodes(G, pos_A, node_size=node_size, node_color='lightblue', alpha=0.5, margins=0.05)
            nx.draw_networkx_labels(G, pos_A, font_size=6, font_weight="bold")
            nx.draw_networkx_edges(G, pos_A, edge_color="black", connectionstyle=connectionstyle, node_size=node_size,
                                   width=2)
            plt.figtext(0.01, 0.02, t.interpret(repo.pretty_term_algebra()), fontsize=14)

            plt.show()
        else:
            print(t)


