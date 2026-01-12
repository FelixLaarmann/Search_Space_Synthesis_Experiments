from cl3s import (SpecificationBuilder, Constructor, Literal, Var, Group, DataGroup,
                  DerivationTree, SearchSpaceSynthesizer)

import torch
import torch.nn as nn
import torch.optim as optim

from synthesis.utils import generate_data

from synthesis.ode_repo import ODE_DAG_Repository

repo = ODE_DAG_Repository(dimensions=[1, 2, 3, 4], linear_feature_dimensions=[1, 2], sharpness_values=[2],
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
                                                                  (repo.Linear(1, 1, True), 1, 1),
                                                                  (repo.Linear(1, 1, True), 1, 1),
                                                                  (repo.Linear(1, 1, True), 1, 1)
                                                              ),  # left, split, right
                                                              (
                                                                  edge,
                                                                  (repo.LTE(0), 1, 2),
                                                                  edge
                                                              ),  # left, gate, right
                                                              (
                                                                  (repo.Product(), 2, 1),
                                                                  (repo.Product(-1), 1, 1),
                                                                  edge
                                                              ),  # left_out, -gate, right
                                                              (
                                                                  edge,
                                                                  (repo.Sum(1), 1, 1),
                                                                  edge
                                                              ),  # left_out, 1-gate, right
                                                              (
                                                                  edge,
                                                                  (repo.Product(), 2, 1)
                                                              ),  # left_out, right_out
                                                              (
                                                                  (repo.Sum(), 2, 1),
                                                              )
                                                          )
                                                      )))
                               & Constructor("Loss", Constructor("type", Literal(repo.MSEloss())))
                               & Constructor("Optimizer", Constructor("type", Literal(repo.Adam(1e-2))))
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
                                     & Constructor("Loss", Constructor("type", Literal(repo.MSEloss())))
                                     & Constructor("Optimizer", Constructor("type", Literal(repo.Adam(1e-2))))
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
                                     & Constructor("Loss", Constructor("type", Literal(repo.MSEloss())))
                                     & Constructor("Optimizer", Constructor("type", Literal(repo.Adam(1e-2))))
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
                                     & Constructor("Loss", Constructor("type", Literal(repo.MSEloss())))
                                     & Constructor("Optimizer", Constructor("type", Literal(repo.Adam(1e-2))))
                                     & Constructor("epochs", Literal(10000))
                                     )

if __name__ == "__main__":

    target = target_from_trapezoid3

    synthesizer = SearchSpaceSynthesizer(repo.specification(), {})

    search_space = synthesizer.construct_search_space(target).prune()
    print("finish synthesis, start enumerate")

    terms = search_space.sample(100, target)

    # term = terms.__next__()

    terms_list = list(terms)

    print("enumeration finished")

    term = terms_list[0]

    print(target)

    print(term.interpret(repo.pytorch_code_algebra()))

    print(f"number of terms: {len(terms_list)}")