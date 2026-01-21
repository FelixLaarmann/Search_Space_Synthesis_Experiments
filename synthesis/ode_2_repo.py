from cl3s import (SpecificationBuilder, Constructor, Literal, Var, Group, DataGroup,
                  DerivationTree, SearchSpaceSynthesizer)
from typing import Any

from dataclasses import dataclass

import networkx as nx

import torch
import torch.nn as nn
import torch.optim as optim

from synthesis.utils import generate_data

class ODE_2_Repository:
    """
            The terms have to be in normal form under the following term rewriting system:

            associativity laws:  (handled with types)

            beside(beside(x,y),z)
            ->
            beside(x, beside(y,z))

            before(before(x,y),z)
            ->
            before(x, before(y,z))

            abiding law:   (handled with types)

            beside(before(m,n,p, w(m,n), x(n,p)), before(m',r,p', y(m',r), z(r,p')))
            ->
            before(m+m', n+r, p+p', beside(w(m,n),y(m',r)), beside(x(n,p),z(r,p')))

            neutrality of edge:   (handled with types)

            before(edge(), x)
            ->
            x

            before(x, edge())
            ->
            x

            swap laws:    (they need to be term predicates :-(...)


            before(swap(m+n, m, n), before(beside(x(n,p), y(m,q)), swap(p+q, p, q)))
            ->
            beside(y(m,q),x(n,p))


            (the three laws below may be handled with types...)
            (one could introduce an "only edges" flag, like ID and non_ID,
            but that would also require a lot of thinking...)
            (nope, that's not possible, because we need at least one term predicate and therefore cannot use
            function types in suffix(), but must abstract with argument() ... this prevents handling these
            laws with types...)

            before(besides(swap(m+n, m, n), copy(p,edge())), besides(copy(n, edge()), swap(m+p, m, p)))
            ->
            swap(m + n + p, m, n+p)

            before(swap(m+n, m, n), swap(n+m, n, m))
            ->
            copy(m+n, edge())

            before(besides(copy(m, edge()), swap(n+p, n, p)), besides(swap(m+p, m, p), copy(n,edge())))
            ->
            swap(m + n + p, m+n, p)


            additionally, we should satisfy the following law (which is implicit in Gibbon's paper...)
            (this is handled with types, too)

            beside(swap(n, 0, n), swap(m, 0, m))
            ->
            swap(n+m, 0, n+m)

            Since we want to synthesize labeled DAGs instead of labeled DAMGs, we also need to ensure that there is
            at most one edge between two nodes. We have to handle this with a term predicate, too.
            The simplest implementation of this predicate will use the edgelist_algebra to check for duplicate edges
            in sequential compositions.
    """

    def __init__(self,
                 linear_feature_dimensions, constant_values, learning_rate_values,
                 n_epoch_values, dimensions=None):
        # additionally to labeled nodes, we have (unlabelled) edges, that needs to be handled additionally
        self.dimensions = range(1, max(linear_feature_dimensions) + 1) if dimensions is None or (max(dimensions) < max(linear_feature_dimensions)) else dimensions
        self.linear_feature_dimensions = linear_feature_dimensions
        self.learning_rate_values = learning_rate_values
        self.n_epoch_values = n_epoch_values
        self.constant_values = constant_values if 1 in constant_values and 0 in constant_values else constant_values + [0, 1]

    @dataclass(frozen=True)
    class Linear:
        in_features: int
        out_features: int
        bias: bool = True

    @dataclass(frozen=True)
    class Sigmoid:
        trivial = ()

    @dataclass(frozen=True)
    class ReLu:
        inplace: bool = False

    @dataclass(frozen=True)
    class Tanh:
        trivial = ()

    @dataclass(frozen=True)
    class LTE:
        threshold: float = 0

    @dataclass(frozen=True)
    class Sum:
        with_constant: float = 0

    @dataclass(frozen=True)
    class Product:
        with_constant: float = 1

    @dataclass(frozen=True)
    class Copy:
        out_dimension: int

    class Label(Group):
        name = "Label"

        def __init__(self, dimensions,
                     linear_feature_dimensions, constant_values):
            self.dimensions = dimensions
            self.linear_feature_dimensions = linear_feature_dimensions
            self.constant_values = constant_values

        def iter_linear(self):
            for in_f in self.linear_feature_dimensions:
                for out_f in self.linear_feature_dimensions:
                    yield ODE_2_Repository.Linear(in_features=in_f, out_features=out_f, bias=True)
                    yield ODE_2_Repository.Linear(in_features=in_f, out_features=out_f, bias=False)

        def iter_sigmoid(self):
            yield ODE_2_Repository.Sigmoid()

        def iter_relu(self):
            yield ODE_2_Repository.ReLu(inplace=False)
            #yield ODE_1_Repository.ReLu(inplace=True)  # inplace not supported here

        def iter_tanh(self):
            yield ODE_2_Repository.Tanh()

        def iter_lte(self):
            for t in self.constant_values:
                yield ODE_2_Repository.LTE(threshold=t)

        def iter_sum(self):
            for c in self.constant_values:
                yield ODE_2_Repository.Sum(c)

        def iter_product(self):
            for c in self.constant_values:
                if c != 0:
                    yield ODE_2_Repository.Product(c)

        def iter_copy(self):
            for o in self.dimensions:
                yield ODE_2_Repository.Copy(out_dimension=o)

        def __iter__(self):
            yield from self.iter_linear()
            yield from self.iter_sigmoid()
            yield from self.iter_relu()
            yield from self.iter_tanh()
            yield from self.iter_lte()
            yield from self.iter_sum()
            yield from self.iter_product()
            yield from self.iter_copy()

        def __contains__(self, item):
            if isinstance(item, ODE_2_Repository.Linear):
                return ((item.in_features in self.linear_feature_dimensions) and
                        (item.out_features in self.linear_feature_dimensions) and
                        (isinstance(item.bias, bool)))
            elif isinstance(item, ODE_2_Repository.Sigmoid):
                return True
            elif isinstance(item, ODE_2_Repository.ReLu):
                return isinstance(item.inplace, bool)
            elif isinstance(item, ODE_2_Repository.Tanh):
                return True
            elif isinstance(item, ODE_2_Repository.LTE):
                return item.threshold in self.constant_values
            elif isinstance(item, ODE_2_Repository.Sum):
                return item.with_constant in self.constant_values
            elif isinstance(item, ODE_2_Repository.Product):
                return item.with_constant in self.constant_values and item.with_constant != 0
            if isinstance(item, ODE_2_Repository.Copy):
                return item.out_dimension in self.dimensions
            else:
                return False

    @dataclass(frozen=True)
    class MSEloss:
        reduction: str = "mean"

    @dataclass(frozen=True)
    class L1Loss:
        reduction: str = "mean"

    class LossFunction(Group):
        name = "Loss_Function"

        def __init__(self):
            pass

        def iter_mseloss(self):
            yield ODE_2_Repository.MSEloss(reduction="mean")
            yield ODE_2_Repository.MSEloss(reduction="sum")
            #yield ODE_1_Repository.MSEloss(reduction="none")

        def iter_l1loss(self):
            yield ODE_2_Repository.L1Loss(reduction="mean")
            yield ODE_2_Repository.L1Loss(reduction="sum")
            #yield ODE_1_Repository.L1Loss(reduction="none")

        def __iter__(self):
            yield from self.iter_mseloss()
            yield from self.iter_l1loss()

        def __contains__(self, value):
            return (value is None or ((isinstance(value, ODE_2_Repository.MSEloss) or isinstance(value, ODE_2_Repository.L1Loss))
                    and (value.reduction in ["mean", "sum", "none"])))

    @dataclass(frozen=True)
    class Adam:
        learning_rate: float = 0.001
        betas: tuple[float, float] = (0.9, 0.999)
        eps: float = 1e-08
        weight_decay: float = 0.0
        amsgrad: bool = False
        maximize: bool = False
        capturable: bool = False
        differentiable: bool = False
        decoupled_weight_decay: bool = False

    class Optimizer(Group):
        name = "Optimizer"

        def __init__(self, learning_rate_values):
            self.learning_rate_values = learning_rate_values

        def __iter__(self):
            for lr in self.learning_rate_values:
                yield ODE_2_Repository.Adam(learning_rate=lr)

        def __contains__(self, value):
            return value is None or (isinstance(value, ODE_1_Repository.Adam) and (value.learning_rate in self.learning_rate_values))

    class Para(Group):
        name = "Para"

        def __init__(self, labels, dimensions):
            self.labels = list(labels) + [None]
            self.dimensions = list(dimensions) + [None]

        def __iter__(self):
            for l in self.labels:
                if l is not None:
                    for i in self.dimensions:
                        if i is not None:
                            for o in self.dimensions:
                                if o is not None:
                                    yield l, i, o
                                    if i == o:
                                        for n in range (0, i):
                                            m = i - n
                                            assert m > 0
                                            yield ("swap", n, m), i, o
            yield None

        def __contains__(self, value):
            return (value is None
                    #or value in ["linear", "sigmoid", "relu", "sharpness_sigmoid", "lte", "sum", "product"]
                    or (isinstance(value, tuple)
                                     and len(value) == 3
                                     and (value[0] in self.labels
                                          or (value[0][0] == "swap"
                                              and (value[0][1] in self.dimensions
                                                   or value[0][1] == 0)
                                              and value[0][2] in self.dimensions))
                                     and value[1] in self.dimensions
                                     and value[2] in self.dimensions
                                     # TODO: something like this? -> and value[1] == value[2] if not value[0] in self.labels and value[0][0] == "swap" else True
                                     ))

    class ParaTuples(Group):
        name = "ParaTuples"

        def __init__(self, para, max_length=3):
            self.para = para
            self.max_length = max_length

        def __iter__(self):
            result = set()

            for n in range(0, self.max_length + 1):
                if n == 0:
                    result.add(())
                else:
                    old_result = result.copy()
                    for para in self.para:
                        for suffix in old_result:
                            result.add((para,) + suffix)
            yield from result

        def __contains__(self, value):
            return value is None or (isinstance(value, tuple) and all(True if v is None else v in self.para for v in value))

        def normalform(self, value) -> bool:
            """
            beside(swap(n, 0, n), swap(m, 0, m))
            ->
            swap(n+m, 0, n+m)
            """
            if value is None:
                return True # because synthesis enforces, that every variance for None will be in normal form
            for l, r in zip(value[:-1], value[1:]):
                if l is not None and r is not None:
                    if len(l) == 3 and l[0] is not None:
                        label, i, o = l
                        if isinstance(label, tuple) and len(label) == 3 and label[0] == "swap" and label[1] == 0:
                            if len(r) == 3 and r[0] is not None:
                                right_label, right_i, right_o = r
                                if isinstance(right_label, tuple) and len(right_label) == 3 and right_label[0] == "swap" and right_label[1] == 0:
                                    """
                                    beside(swap(n, 0, n), swap(m, 0, m))
                                    ->
                                    swap(n+m, 0, n+m)
                                    """
                                    return False
            return True

        def normalize(self, value):
            while(not self.normalform(value)):
                new_value = value
                index = 0 # index in new_value
                for l, r in zip(value[:-1], value[1:]):
                    if len(l) == 3:
                        label, i, o = l
                        if isinstance(label, tuple) and len(label) == 3 and label[0] == "swap" and label[1] == 0:
                            n = label[2]
                            if len(r) == 3 and r[0] is not None:
                                right_label, right_i, right_o = r
                                if (isinstance(right_label, tuple) and len(right_label) == 3 and
                                        right_label[0] == "swap" and right_label[1] == 0):
                                    m = right_label[2]
                                    """
                                    beside(swap(n, 0, n), swap(m, 0, m))
                                    ->
                                    swap(n+m, 0, n+m)
                                    """
                                    # i < len(new_value) is an invariant, because len(zip(value[:-1], value[1:])) == len(value) - 1
                                    before_i = new_value[:index]
                                    after_i = new_value[index+2:]
                                    new_value = before_i + ((("swap", 0, n + m), n + m, n + m),) + after_i
                                    break
                    index += 1
                value = new_value
            return value

    class ParaTupleTuples(Group):
        name = "ParaTupleTuples"

        def __init__(self, para_tuples):
            self.para_tuples = para_tuples

        def __iter__(self):
            return super().__iter__()

        def __contains__(self, value):
            return value is None or (isinstance(value, tuple) and all(True if v is None else v in self.para_tuples for v in value))

        def normalform(self, value) -> bool:
            """
            The associativity laws are handled by the way we use python tuples.
            The abiding law is an invariance, because otherwise we couldn't use tuples of tuples.

            Therefore, we only need to check:
            - Neutrality of edges
            - Swap laws
            - Unique representation of parallel edges (swaps with n=0) (handled in ParaTuples)
            """
            if value is None:
                return True # because synthesis enforces, that every variance for None will be in normal form
            for l, r in zip(value[:-1], value[1:]):
                if l is not None and r is not None:
                    if not (self.para_tuples.normalform(l) and self.para_tuples.normalform(r)):
                        return False
                    if len(l) == 1 and l[0] is not None:
                        label, i, o = l[0]
                        if isinstance(label, tuple) and len(label) == 3 and label[0] == "swap":
                            m = label[1]
                            n = label[2]
                            if m == 0:
                                """
                                before(edge(), x)
                                ->
                                x
                                """
                                return False
                            if len(r) == 1 and r[0] is not None:
                                """
                                before(swap(m+n, m, n), swap(n+m, n, m))
                                ->
                                copy(m+n, edge())
                                """
                                right_label, right_i, right_o = r[0]
                                if isinstance(right_label, tuple) and len(right_label) == 3 and right_label[0] == "swap":
                                    right_m = right_label[1]
                                    right_n = right_label[2]
                                    if right_m is not None and right_n is not None:
                                        if m == right_n and n == right_m:
                                            return False
                    if len(r) == 1 and r[0] is not None:
                        """
                        before(x, edge())
                        ->
                        x
                        """
                        label, i, o = r[0]
                        if isinstance(label, tuple) and len(label) == 3 and label[0] == "swap":
                            if label[1] == 0:
                                return False
                    if len(l) == 2 and len(r) == 2:
                        left_first, left_second = l
                        right_first, right_second = r
                        if left_first is not None and left_second is not None and right_first is not None and right_second is not None:
                            label_l_1, i_l_1, o_l_1 = left_first
                            label_l_2, i_l_2, o_l_2 = left_second
                            label_r_1, i_r_1, o_r_1 = right_first
                            label_r_2, i_r_2, o_r_2 = right_second
                            if isinstance(label_l_1, tuple) and len(label_l_1) == 3 and label_l_1[0] == "swap":
                                m = label_l_1[1]
                                n = label_l_1[2]
                                if isinstance(label_l_2, tuple) and len(label_l_2) == 3 and label_l_2[0] == "swap" and label_l_2[1] == 0:
                                    p = label_l_2[2]
                                    if isinstance(label_r_1, tuple) and len(label_r_1) == 3 and label_r_1[0] == "swap" and label_r_1[1] == 0:
                                        right_n = label_r_1[2]
                                        if isinstance(label_r_2, tuple) and len(label_r_2) == 3 and label_r_2[0] == "swap":
                                            right_m = label_r_2[1]
                                            right_p = label_r_2[2]
                                            if m is not None and n is not None and p is not None and right_m is not None and right_n is not None and right_p is not None:
                                                if m == right_m and n == right_n and p == right_p:
                                                    """
                                                    before(besides(swap(m+n, m, n), copy(p,edge())), besides(copy(n, edge()), swap(m+p, m, p)))
                                                    ->
                                                    swap(m + n + p, m, n+p)
                                                    """
                                                    return False
                            if isinstance(label_l_1, tuple) and len(label_l_1) == 3 and label_l_1[0] == "swap" and label_l_1[1] == 0:
                                m = label_l_1[2]
                                if isinstance(label_l_2, tuple) and len(label_l_2) == 3 and label_l_2[0] == "swap":
                                    n = label_l_2[1]
                                    p = label_l_2[2]
                                    if isinstance(label_r_1, tuple) and len(label_r_1) == 3 and label_r_1[0] == "swap":
                                        right_m = label_r_1[1]
                                        right_p = label_r_1[2]
                                        if isinstance(label_r_2, tuple) and len(label_r_2) == 3 and label_r_2[0] == "swap" and label_r_2[1] == 0:
                                            right_n = label_r_2[2]
                                            if m is not None and n is not None and p is not None and right_m is not None and right_n is not None and right_p is not None:
                                                if m == right_m and n == right_n and p == right_p:
                                                    """
                                                    before(besides(copy(m, edge()), swap(n+p, n, p)), besides(swap(m+p, m, p), copy(n,edge())))
                                                    ->
                                                    swap(m + n + p, m+n, p)
                                                    """
                                                    return False

            for l, m, r in zip(value[:-2], value[1:-1], value[2:]):
                if l is not None and m is not None and r is not None:
                    if len(l) == 1 and l[0] is not None:
                        left_label, left_i, left_o = l[0]
                        if isinstance(left_label, tuple) and len(left_label) == 3 and left_label[0] == "swap":
                            left_m = left_label[1]
                            left_n = left_label[2]
                            if len(m) == 2 and len(r) == 1 and r[0] is not None:
                                mid_first, mid_second = m
                                right_label, right_i, right_o = r[0]
                                if mid_first is not None and mid_second is not None:
                                    mid_first_label, mid_n, mid_p = mid_first
                                    mid_second_label, mid_m, mid_q = mid_second
                                    if isinstance(right_label, tuple) and len(right_label) == 3 and right_label[0] == "swap":
                                        right_p = right_label[1]
                                        right_q = right_label[2]
                                        if left_m is not None and left_n is not None and mid_m is not None and mid_n is not None and mid_p is not None and mid_q is not None and right_p is not None and right_q is not None:
                                            if left_m == mid_m and left_n == mid_n and right_p == mid_p and right_q == mid_q:
                                                """
                                                before(swap(m + n, m, n), before(beside(x(n, p), y(m, q)), swap(p + q, p, q)))
                                                ->
                                                beside(y(m, q), x(n, p))
                                                """
                                                return False
            # TODO: DAG Criteria: At most one edge between two nodes.
            return True

        def normalize(self, value):
            if (not self.normalform(value)):
                value = tuple(map(self.para_tuples.normalize, value))
            while(not self.normalform(value)):
                new_value = value
                index = 0 # index in new_value
                for l, r in zip(value[:-1], value[1:]):
                    if len(l) == 1:
                        label, i, o = l[0]
                        if isinstance(label, tuple) and len(label) == 3 and label[0] == "swap":
                            m = label[1]
                            n = label[2]
                            if m == 0:
                                """
                                before(edge(), x)
                                ->
                                x
                                """
                                # i < len(new_value) is an invariant, because len(zip(value[:-1], value[1:])) == len(value) - 1
                                before_i = new_value[:index]
                                after_i = new_value[index + 2:]
                                new_value = before_i + (r,) + after_i
                                break
                            if len(r) == 1:
                                """
                                before(swap(m+n, m, n), swap(n+m, n, m))
                                ->
                                copy(m+n, edge())
                                """
                                right_label, right_i, right_o = r[0]
                                if (isinstance(right_label, tuple) and len(right_label) == 3
                                        and right_label[0] == "swap"):
                                    right_m = right_label[1]
                                    right_n = right_label[2]
                                    if m == right_n and n == right_m:
                                        before_i = new_value[:index]
                                        after_i = new_value[index + 2:]
                                        new_value = before_i + (((("swap", 0, n + m), n + m, n + m),),) + after_i
                                        break
                    if len(r) == 1 and r[0] is not None:
                        """
                        before(x, edge())
                        ->
                        x
                        """
                        label, i, o = r[0]
                        if isinstance(label, tuple) and len(label) == 3 and label[0] == "swap":
                            if label[1] == 0:
                                before_i = new_value[:index]
                                after_i = new_value[index + 2:]
                                new_value = before_i + (l,) + after_i
                                break
                    if len(l) == 2 and len(r) == 2:
                        left_first, left_second = l
                        right_first, right_second = r
                        if left_first and left_second and right_first and right_second:
                            label_l_1, i_l_1, o_l_1 = left_first
                            label_l_2, i_l_2, o_l_2 = left_second
                            label_r_1, i_r_1, o_r_1 = right_first
                            label_r_2, i_r_2, o_r_2 = right_second
                            if isinstance(label_l_1, tuple) and len(label_l_1) == 3 and label_l_1[0] == "swap":
                                m = label_l_1[1]
                                n = label_l_1[2]
                                if (isinstance(label_l_2, tuple) and len(label_l_2) == 3 and label_l_2[0] == "swap"
                                        and label_l_2[1] == 0):
                                    p = label_l_2[2]
                                    if (isinstance(label_r_1, tuple) and len(label_r_1) == 3 and label_r_1[0] == "swap"
                                            and label_r_1[1] == 0):
                                        right_n = label_r_1[2]
                                        if (isinstance(label_r_2, tuple) and len(label_r_2) == 3 and
                                                label_r_2[0] == "swap"):
                                            right_m = label_r_2[1]
                                            right_p = label_r_2[2]
                                            if m == right_m and n == right_n and p == right_p:
                                                """
                                                before(besides(swap(m+n, m, n), copy(p,edge())), besides(copy(n, edge()), swap(m+p, m, p)))
                                                ->
                                                swap(m + n + p, m, n+p)
                                                """
                                                before_i = new_value[:index]
                                                after_i = new_value[index + 2:]
                                                new_value = before_i + (((("swap", m, n + p), m + n + p, m + n + p),),) + after_i
                                                break
                            if (isinstance(label_l_1, tuple) and len(label_l_1) == 3 and label_l_1[0] == "swap" and
                                    label_l_1[1] == 0):
                                m = label_l_1[2]
                                if isinstance(label_l_2, tuple) and len(label_l_2) == 3 and label_l_2[0] == "swap":
                                    n = label_l_2[1]
                                    p = label_l_2[2]
                                    if isinstance(label_r_1, tuple) and len(label_r_1) == 3 and label_r_1[0] == "swap":
                                        right_m = label_r_1[1]
                                        right_p = label_r_1[2]
                                        if (isinstance(label_r_2, tuple) and len(label_r_2) == 3 and
                                                label_r_2[0] == "swap" and label_r_2[1] == 0):
                                            right_n = label_r_2[2]
                                            if m == right_m and n == right_n and p == right_p:
                                                """
                                                before(besides(copy(m, edge()), swap(n+p, n, p)), besides(swap(m+p, m, p), copy(n,edge())))
                                                ->
                                                swap(m + n + p, m+n, p)
                                                """
                                                before_i = new_value[:index]
                                                after_i = new_value[index + 2:]
                                                new_value = before_i + (((("swap", m + n, p), m + n + p, m + n + p),),) + after_i
                                                break
                for l, m, r in zip(value[:-2], value[1:-1], value[2:]):
                    if len(l) == 1:
                        left_label, left_i, left_o = l[0]
                        if isinstance(left_label, tuple) and len(left_label) == 3 and left_label[0] == "swap":
                            left_m = left_label[1]
                            left_n = left_label[2]
                            if len(m) == 2 and len(r) == 1:
                                mid_first, mid_second = m
                                right_label, right_i, right_o = r[0]
                                mid_first_label, mid_n, mid_p = mid_first
                                mid_second_label, mid_m, mid_q = mid_second
                                if isinstance(right_label, tuple) and len(right_label) == 3 and right_label[0] == "swap":
                                    right_p = right_label[1]
                                    right_q = right_label[2]
                                    if left_m == mid_m and left_n == mid_n and right_p == mid_p and right_q == mid_q:
                                        """
                                        before(swap(m + n, m, n), before(beside(x(n, p), y(m, q)), swap(p + q, p, q)))
                                        ->
                                        beside(y(m, q), x(n, p))
                                        """
                                        before_i = new_value[:index]
                                        after_i = new_value[index + 3:]
                                        new_value = before_i + ((mid_second, mid_first),) + after_i
                                        break
                value = new_value
                # TODO: How to handel DAG Criteria in normalization?
            return value

    @staticmethod
    def swaplaw1(head: DerivationTree[Any, str, Any], tail: DerivationTree[Any, str, Any]) -> bool:
        """
        before(swap(m+n, m, n), before(beside(x(n,p), y(m,q)), swap(p+q, p, q)))
        ->
        beside(y(m,q),x(n,p))

        forbid the pattern on the left-hand side of the rewrite rule by returning False if it is matched
        """

        left = head.root
        right = tail.root

        if "beside_singleton" in left and "before_cons" in right:
            if len(head.children) != 5 or len(tail.children) != 9:
                raise ValueError("Derivation trees have not the expected shape.")
            left_term = head.children[4]
            right_head = tail.children[7]
            right_tail = tail.children[8]

            left_term_root = left_term.root
            right_head_root = right_head.root
            right_tail_root = right_tail.root
            if (
                    left_term_root == "swap") and "beside_cons" in right_head_root and "before_singleton" in right_tail_root:
                if len(left_term.children) != 19 or len(right_head.children) != 11 or len(right_tail.children) != 6:
                    raise ValueError("Derivation trees have not the expected shape.")
                m = left_term.children[1]
                n = left_term.children[2]
                x_n = right_head.children[1]
                x_p = right_head.children[4]
                right_head_tail = right_head.children[10]
                right_tail_term = right_tail.children[5]
                right_head_tail_root = right_head_tail.root
                right_tail_term_root = right_tail_term.root
                if "beside_singleton" in right_head_tail_root and "beside_singleton" in right_tail_term_root:
                    if len(right_head_tail.children) != 5 or len(right_tail_term.children) != 5:
                        raise ValueError("Derivation trees have not the expected shape.")
                    y_m = right_head_tail.children[0]
                    y_q = right_head_tail.children[1]
                    right_swap = right_tail_term.children[4]
                    right_swap_root = right_swap.root
                    if right_swap_root == "swap":
                        if len(right_swap.children) != 19:
                            raise ValueError("Derivation trees have not the expected shape.")
                        p = right_swap.children[1]
                        q = right_swap.children[2]
                        if m == y_m and n == x_n and p == x_p and q == y_q:
                            return False
            elif (left == "swap") and "beside_cons" in right_head_root and "before_cons" in right_tail_root:
                if len(left_term.children) != 19 or len(right_head.children) != 11 or len(right_tail.children) != 9:
                    raise ValueError("Derivation trees have not the expected shape.")
                m = left_term.children[1]
                n = left_term.children[2]
                x_n = right_head.children[1]
                x_p = right_head.children[4]
                right_head_tail = right_head.children[10]
                right_tail_head = right_tail.children[7]
                right_head_tail_root = right_head_tail.root
                right_tail_term_root = right_tail_head.root
                if "beside_singleton" in right_head_tail_root and "beside_singleton" in right_tail_term_root:
                    if len(right_head_tail.children) != 5 or len(right_tail_head.children) != 5:
                        raise ValueError("Derivation trees have not the expected shape.")
                    y_m = right_head_tail.children[0]
                    y_q = right_head_tail.children[1]
                    right_swap = right_tail_head.children[4]
                    right_swap_root = right_swap.root
                    if right_swap_root == "swap":
                        if len(right_swap.children) != 19:
                            raise ValueError("Derivation trees have not the expected shape.")
                        p = right_swap.children[1]
                        q = right_swap.children[2]
                        if m == y_m and n == x_n and p == x_p and q == y_q:
                            return False
        return True

    @staticmethod
    def swaplaw2(head: DerivationTree[Any, str, Any], tail: DerivationTree[Any, str, Any]) -> bool:
        """
        before(besides(swap(m+n, m, n), copy(p,edge())), besides(copy(n, edge()), swap(m+p, m, p)))
        ->
        swap(m + n + p, m, n+p)

        forbid the pattern on the left-hand side of the rewrite rule by returning False if it is matched

        """

        left = head.root
        right = tail.root

        if "beside_cons" in left and "before_singleton" in right:
            if len(head.children) != 11 or len(tail.children) != 6:
                raise ValueError("Derivation trees have not the expected shape.")
            left_head = head.children[9]
            left_tail = head.children[10]
            right_term = tail.children[5]

            left_swap = left_head.root
            left_tail_root = left_tail.root
            right_term_root = right_term.root
            if (left_swap == "swap") and "beside_singleton" in left_tail_root and "beside_cons" in right_term_root:
                if len(left_head.children) != 19 or len(left_tail.children) != 5 or len(right_term.children) != 11:
                    raise ValueError("Derivation trees have not the expected shape.")
                m = left_head.children[1]
                n = left_head.children[2]
                left_tail_term = left_tail.children[4]  # swap(p, 0, p)
                right_head = right_term.children[9]  # swap(n, 0, n)
                right_tail = right_term.children[10]

                left_tail_swap = left_tail_term.root
                right_head_root = right_head.root
                right_tail_root = right_tail.root
                if left_tail_swap == "edges" and right_head_root == "edges" and "beside_singleton" in right_tail_root:
                    if len(left_tail_term.children) != 17 or len(right_head.children) != 17 or len(
                            right_tail.children) != 5:
                        raise ValueError("Derivation trees have not the expected shape.")
                    p = left_tail_term.children[0]
                    right_n = right_head.children[0]

                    right_tail_term = right_tail.children[4]  # swap(m+p, m, p)
                    right_tail_term_root = right_tail_term.root
                    if (right_tail_term_root == "swap") and n == right_n:
                        if len(right_tail_term.children) != 19:
                            raise ValueError("Derivation trees have not the expected shape.")
                        right_m = right_tail_term.children[1]
                        right_p = right_tail_term.children[2]
                        if m == right_m and p == right_p:
                            return False
        elif "beside_cons" in left and "before_cons" in right:
            if len(head.children) != 11 or len(tail.children) != 9:
                raise ValueError("Derivation trees have not the expected shape.")
            left_head = head.children[9]
            left_tail = head.children[10]
            right_term = tail.children[7]

            left_swap = left_head.root
            left_tail_root = left_tail.root
            right_term_root = right_term.root
            if (left_swap == "swap") and "beside_singleton" in left_tail_root and "beside_cons" in right_term_root:
                if len(left_head.children) != 19 or len(left_tail.children) != 5 or len(right_term.children) != 11:
                    raise ValueError("Derivation trees have not the expected shape.")
                m = left_head.children[1]
                n = left_head.children[2]
                left_tail_term = left_tail.children[4]  # swap(p, 0, p)
                right_head = right_term.children[9]  # swap(n, 0, n)
                right_tail = right_term.children[10]

                left_tail_swap = left_tail_term.root
                right_head_root = right_head.root
                right_tail_root = right_tail.root
                if left_tail_swap == "edges" and right_head_root == "edges" and "beside_singleton" in right_tail_root:
                    if len(left_tail_term.children) != 17 or len(right_head.children) != 17 or len(
                            right_tail.children) != 5:
                        raise ValueError("Derivation trees have not the expected shape.")
                    p = left_tail_term.children[0]
                    right_n = right_head.children[0]

                    right_tail_term = right_tail.children[4]  # swap(m+p, m, p)
                    right_tail_term_root = right_tail_term.root
                    if right_tail_term_root == "swap" and n == right_n:
                        if len(right_tail_term.children) != 19:
                            raise ValueError("Derivation trees have not the expected shape.")
                        right_m = right_tail_term.children[1]
                        right_p = right_tail_term.children[2]
                        if m == right_m and p == right_p:
                            return False
        return True

    @staticmethod
    def swaplaw3(head: DerivationTree[Any, str, Any], tail: DerivationTree[Any, str, Any]) -> bool:
        """
        before(swap(m+n, m, n), swap(n+m, n, m))
        ->
        copy(m+n, edge())

        forbid the pattern on the left-hand side of the rewrite rule by returning False if it is matched
        """
        left = head.root
        right = tail.root

        if "beside_singleton" in left and "before_singleton" in right:
            if len(head.children) != 5 or len(tail.children) != 6:
                raise ValueError("Derivation trees have not the expected shape.")
            left_term = head.children[4]
            right_term = tail.children[5]

            left_term_root = left_term.root
            right_term_root = right_term.root
            if (left_term_root == "swap") and "beside_singleton" in right_term_root:
                if len(left_term.children) != 19 or len(right_term.children) != 5:
                    raise ValueError("Derivation trees have not the expected shape.")
                m = left_term.children[1]
                n = left_term.children[2]

                right_beside = right_term.children[4]
                right_beside_root = right_beside.root
                if right_beside_root == "swap" or right_beside_root == "swap":
                    if len(right_beside.children) != 19:
                        raise ValueError("Derivation trees have not the expected shape.")
                    right_n = right_beside.children[1]
                    right_m = right_beside.children[2]
                    if m == right_m and n == right_n:
                        return False
        elif "beside_singleton" in left and "before_cons" in right:
            if len(head.children) != 5 or len(tail.children) != 9:
                raise ValueError("Derivation trees have not the expected shape.")
            left_term = head.children[4]
            right_term = tail.children[7]

            left_term_root = left_term.root
            right_term_root = right_term.root
            if (left_term_root == "swap") and "beside_singleton" in right_term_root:
                if len(left_term.children) != 19 or len(right_term.children) != 5:
                    raise ValueError("Derivation trees have not the expected shape.")
                m = left_term.children[1]
                n = left_term.children[2]

                right_beside = right_term.children[4]
                right_beside_root = right_beside.root
                if right_beside_root == "swap":
                    if len(right_beside.children) != 19:
                        raise ValueError("Derivation trees have not the expected shape.")
                    right_n = right_beside.children[1]
                    right_m = right_beside.children[2]
                    if m == right_m and n == right_n:
                        return False
        return True

    @staticmethod
    def swaplaw4(head: DerivationTree[Any, str, Any], tail: DerivationTree[Any, str, Any]) -> bool:
        """
        before(besides(copy(m, edge()), swap(n+p, n, p)), besides(swap(m+p, m, p), copy(n,edge())))
        ->
        swap(m + n + p, m+n, p)

        forbid the pattern on the left-hand side of the rewrite rule by returning False if it is matched
        """
        left = head.root
        right = tail.root

        if "beside_cons" in left and "before_singleton" in right:
            if len(head.children) != 11 or len(tail.children) != 6:
                raise ValueError("Derivation trees have not the expected shape.")
            left_head = head.children[9]
            left_tail = head.children[10]
            right_term = tail.children[5]

            left_head_root = left_head.root
            left_tail_root = left_tail.root
            right_term_root = right_term.root
            if left_head_root == "edges" and "beside_singleton" in left_tail_root and "beside_cons" in right_term_root:
                if len(left_head.children) != 17 or len(left_tail.children) != 5 or len(right_term.children) != 11:
                    raise ValueError("Derivation trees have not the expected shape.")
                m = left_head.children[0]

                left_tail_term = left_tail.children[4]
                right_head = right_term.children[9]
                right_tail = right_term.children[10]

                left_tail_term_root = left_tail_term.root
                right_head_root = right_head.root
                right_tail_root = right_tail.root
                if left_tail_term_root == "swap" and right_head_root == "swap" and "beside_singleton" in right_tail_root:
                    if len(left_tail_term.children) != 19 or len(right_head.children) != 19 or len(
                            right_tail.children) != 5:
                        raise ValueError("Derivation trees have not the expected shape.")
                    n = left_tail_term.children[1]
                    p = left_tail_term.children[2]

                    right_m = right_head.children[1]
                    right_p = right_head.children[2]

                    right_tail_term = right_tail.children[4]
                    right_tail_term_root = right_tail_term.root

                    if right_tail_term_root == "edges" and m == right_m and p == right_p:
                        if len(right_tail_term.children) != 17:
                            raise ValueError("Derivation trees have not the expected shape.")
                        right_n = right_tail_term.children[0]
                        if n == right_n:
                            return False

        elif "beside_cons" in left and "before_cons" in right:
            if len(head.children) != 11 or len(tail.children) != 9:
                raise ValueError("Derivation trees have not the expected shape.")
            left_head = head.children[9]
            left_tail = head.children[10]
            right_term = tail.children[7]

            left_head_root = left_head.root
            left_tail_root = left_tail.root
            right_term_root = right_term.root
            if left_head_root == "edges" and "beside_singleton" in left_tail_root and "beside_cons" in right_term_root:
                if len(left_head.children) != 17 or len(left_tail.children) != 5 or len(right_term.children) != 11:
                    raise ValueError("Derivation trees have not the expected shape.")
                m = left_head.children[0]

                left_tail_term = left_tail.children[4]
                right_head = right_term.children[9]
                right_tail = right_term.children[10]

                left_tail_term_root = left_tail_term.root
                right_head_root = right_head.root
                right_tail_root = right_tail.root
                if left_tail_term_root == "swap" and right_head_root == "swap" and "beside_singleton" in right_tail_root:
                    if len(left_tail_term.children) != 19 or len(right_head.children) != 19 or len(
                            right_tail.children) != 5:
                        raise ValueError("Derivation trees have not the expected shape.")
                    n = left_tail_term.children[1]
                    p = left_tail_term.children[2]

                    right_m = right_head.children[1]
                    right_p = right_head.children[2]

                    right_tail_term = right_tail.children[4]
                    right_tail_term_root = right_tail_term.root

                    if right_tail_term_root == "edges" and m == right_m and p == right_p:
                        if len(right_tail_term.children) != 17:
                            raise ValueError("Derivation trees have not the expected shape.")
                        right_n = right_tail_term.children[0]
                        if n == right_n:
                            return False
        return True


    def specification(self):
        labels = self.Label(self.dimensions, self.linear_feature_dimensions, self.constant_values)
        para_labels = self.Para(labels, self.dimensions)
        #print("linear" in para_labels)
        paratuples = self.ParaTuples(para_labels, max_length=max(self.dimensions))
        paratupletuples = self.ParaTupleTuples(paratuples)
        dimension = DataGroup("dimension", self.dimensions)
        dimension_with_None = DataGroup("dimension_with_None", list(self.dimensions) + [None])
        #feature_dimension = DataGroup("feature_dimension", self.linear_feature_dimensions)
        loss_function = self.LossFunction()
        optimizer = self.Optimizer(self.learning_rate_values)
        epochs = DataGroup("epochs", self.n_epoch_values)

        return {
            # atomic components are nodes and edges
            # but edges are a special case of swaps (swaps, that do not change anything)
            # so we can just define edges as a special case of swaps here
            #"edge": Constructor("DAG", Constructor("input", Literal(1))
            #                    & Constructor("input", Literal(None))
            #                    & Constructor("output", Literal(1))
            #                    & Constructor("output", Literal(None))
            #                    & Constructor("structure", Literal(((("swap", Literal(0), Literal(1)),),)))
            #                    & Constructor("structure", Literal(((None,),)))),
            # (m parallel) edges are swaps with n=0
            "edges": SpecificationBuilder()
            .parameter("io", dimension)
            .parameter("para1", para_labels, lambda v: [(("swap", 0, v["io"]), v["io"], v["io"])])
            .parameter("para2", para_labels, lambda v: [(("swap", 0, None), v["io"], v["io"])])
            .parameter("para3", para_labels, lambda v: [(("swap", None, v["io"]), v["io"], v["io"])])
            .parameter("para4", para_labels, lambda v: [(("swap", None, None), v["io"], v["io"])])
            .parameter("para5", para_labels, lambda v: [(("swap", 0, v["io"]), v["io"], None)])
            .parameter("para6", para_labels, lambda v: [(("swap", 0, None), v["io"], None)])
            .parameter("para7", para_labels, lambda v: [(("swap", None, v["io"]), v["io"], None)])
            .parameter("para8", para_labels, lambda v: [(("swap", None, None), v["io"], None)])
            .parameter("para9", para_labels, lambda v: [(("swap", 0, v["io"]), None, v["io"])])
            .parameter("para10", para_labels, lambda v: [(("swap", 0, None), None, v["io"])])
            .parameter("para11", para_labels, lambda v: [(("swap", None, v["io"]), None, v["io"])])
            .parameter("para12", para_labels, lambda v: [(("swap", None, None), None, v["io"])])
            .parameter("para13", para_labels, lambda v: [(("swap", 0, v["io"]), None, None)])
            .parameter("para14", para_labels, lambda v: [(("swap", 0, None), None, None)])
            .parameter("para15", para_labels, lambda v: [(("swap", None, v["io"]), None, None)])
            .parameter("para16", para_labels, lambda v: [(("swap", None, None), None, None)])
            .suffix(Constructor("DAG_component", Constructor("input", Var("io"))
                                & Constructor("input", Literal(None))
                                & Constructor("output", Var("io"))
                                & Constructor("output", Literal(None))
                                & Constructor("structure", Var("para1"))
                                & Constructor("structure", Var("para2"))
                                & Constructor("structure", Var("para3"))
                                & Constructor("structure", Var("para4"))
                                & Constructor("structure", Var("para5"))
                                & Constructor("structure", Var("para6"))
                                & Constructor("structure", Var("para7"))
                                & Constructor("structure", Var("para8"))
                                & Constructor("structure", Var("para9"))
                                & Constructor("structure", Var("para10"))
                                & Constructor("structure", Var("para11"))
                                & Constructor("structure", Var("para12"))
                                & Constructor("structure", Var("para13"))
                                & Constructor("structure", Var("para14"))
                                & Constructor("structure", Var("para15"))
                                & Constructor("structure", Var("para16"))
                                & Constructor("structure", Literal(None))
                                ) & Constructor("ID")
                    ),

            "swap": SpecificationBuilder()
            .parameter("io", dimension)
            .parameter("n", dimension, lambda v: range(1, v["io"]))
            .parameter("m", dimension, lambda v: [v["io"] - v["n"]]) # m > 0
            .parameter("para1", para_labels, lambda v: [(("swap", v["n"], v["m"]), v["io"], v["io"])])
            .parameter("para2", para_labels, lambda v: [(("swap", v["n"], None), v["io"], v["io"])])
            .parameter("para3", para_labels, lambda v: [(("swap", None, v["m"]), v["io"], v["io"])])
            .parameter("para4", para_labels, lambda v: [(("swap", None, None), v["io"], v["io"])])
            .parameter("para5", para_labels, lambda v: [(("swap", v["n"], v["m"]), v["io"], None)])
            .parameter("para6", para_labels, lambda v: [(("swap", v["n"], None), v["io"], None)])
            .parameter("para7", para_labels, lambda v: [(("swap", None, v["m"]), v["io"], None)])
            .parameter("para8", para_labels, lambda v: [(("swap", None, None), v["io"], None)])
            .parameter("para9", para_labels, lambda v: [(("swap", v["n"], v["m"]), None, v["io"])])
            .parameter("para10", para_labels, lambda v: [(("swap", v["n"], None), None, v["io"])])
            .parameter("para11", para_labels, lambda v: [(("swap", None, v["m"]), None, v["io"])])
            .parameter("para12", para_labels, lambda v: [(("swap", None, None), None, v["io"])])
            .parameter("para13", para_labels, lambda v: [(("swap", v["n"], v["m"]), None, None)])
            .parameter("para14", para_labels, lambda v: [(("swap", v["n"], None), None, None)])
            .parameter("para15", para_labels, lambda v: [(("swap", None, v["m"]), None, None)])
            .parameter("para16", para_labels, lambda v: [(("swap", None, None), None, None)])
            .suffix(Constructor("DAG_component", Constructor("input", Var("io"))
                                & Constructor("input", Literal(None))
                                & Constructor("output", Var("io"))
                                & Constructor("output", Literal(None))
                                & Constructor("structure", Var("para1"))
                                & Constructor("structure", Var("para2"))
                                & Constructor("structure", Var("para3"))
                                & Constructor("structure", Var("para4"))
                                & Constructor("structure", Var("para5"))
                                & Constructor("structure", Var("para6"))
                                & Constructor("structure", Var("para7"))
                                & Constructor("structure", Var("para8"))
                                & Constructor("structure", Var("para9"))
                                & Constructor("structure", Var("para10"))
                                & Constructor("structure", Var("para11"))
                                & Constructor("structure", Var("para12"))
                                & Constructor("structure", Var("para13"))
                                & Constructor("structure", Var("para14"))
                                & Constructor("structure", Var("para15"))
                                & Constructor("structure", Var("para16"))
                                & Constructor("structure", Literal(None))
                                ) & Constructor("non_ID")
                    ),

            "linear_layer": SpecificationBuilder()
            .parameter("l", labels, lambda v: list(labels.iter_linear()))
            .parameter("i", dimension, lambda v: [v["l"].in_features])
            .parameter("o", dimension, lambda v: [v["l"].out_features])
            .parameter("para1", para_labels, lambda v: [(v["l"], v["i"], v["o"])])
            .parameter("para2", para_labels, lambda v: [(v["l"], v["i"], None)])
            .parameter("para3", para_labels, lambda v: [(v["l"], None, v["o"])])
            .parameter("para4", para_labels, lambda v: [(None, v["i"], v["o"])])
            .parameter("para5", para_labels, lambda v: [(v["l"], None, None)])
            .parameter("para6", para_labels, lambda v: [(None, None, v["o"])])
            .parameter("para7", para_labels, lambda v: [(None, v["i"], None)])
            .suffix(Constructor("DAG_component",
                                Constructor("input", Var("i"))
                                & Constructor("input", Literal(None))
                                & Constructor("output", Var("o"))
                                & Constructor("output", Literal(None))
                                & Constructor("structure", Var("para1"))
                                & Constructor("structure", Var("para2"))
                                & Constructor("structure", Var("para3"))
                                & Constructor("structure", Var("para4"))
                                & Constructor("structure", Var("para5"))
                                & Constructor("structure", Var("para6"))
                                & Constructor("structure", Var("para7"))
                                #& Constructor("structure", Literal("linear"))
                                & Constructor("structure", Literal(None))
                                ) & Constructor("non_ID")
                    ),

            "sigmoid": SpecificationBuilder()
            .parameter("l", labels, lambda v: list(labels.iter_sigmoid()))
            .parameter("i", dimension)
            .parameter("o", dimension, lambda v: [v["i"]])
            .parameter("para1", para_labels, lambda v: [(v["l"], v["i"], v["o"])])
            .parameter("para2", para_labels, lambda v: [(v["l"], v["i"], None)])
            .parameter("para3", para_labels, lambda v: [(v["l"], None, v["o"])])
            .parameter("para4", para_labels, lambda v: [(None, v["i"], v["o"])])
            .parameter("para5", para_labels, lambda v: [(v["l"], None, None)])
            .parameter("para6", para_labels, lambda v: [(None, None, v["o"])])
            .parameter("para7", para_labels, lambda v: [(None, v["i"], None)])
            .suffix(Constructor("DAG_component",
                                Constructor("input", Var("i"))
                                & Constructor("input", Literal(None))
                                & Constructor("output", Var("o"))
                                & Constructor("output", Literal(None))
                                & Constructor("structure", Var("para1"))
                                & Constructor("structure", Var("para2"))
                                & Constructor("structure", Var("para3"))
                                & Constructor("structure", Var("para4"))
                                & Constructor("structure", Var("para5"))
                                & Constructor("structure", Var("para6"))
                                & Constructor("structure", Var("para7"))
                                #& Constructor("structure", Literal("sigmoid"))
                                & Constructor("structure", Literal(None))
                                ) & Constructor("non_ID")
                    ),

            "relu": SpecificationBuilder()
            .parameter("l", labels, lambda v: list(labels.iter_relu()))
            .parameter("i", dimension)
            .parameter("o", dimension, lambda v: [v["i"]])
            .parameter("para1", para_labels, lambda v: [(v["l"], v["i"], v["o"])])
            .parameter("para2", para_labels, lambda v: [(v["l"], v["i"], None)])
            .parameter("para3", para_labels, lambda v: [(v["l"], None, v["o"])])
            .parameter("para4", para_labels, lambda v: [(None, v["i"], v["o"])])
            .parameter("para5", para_labels, lambda v: [(v["l"], None, None)])
            .parameter("para6", para_labels, lambda v: [(None, None, v["o"])])
            .parameter("para7", para_labels, lambda v: [(None, v["i"], None)])
            .suffix(Constructor("DAG_component",
                                Constructor("input", Var("i"))
                                & Constructor("input", Literal(None))
                                & Constructor("output", Var("o"))
                                & Constructor("output", Literal(None))
                                & Constructor("structure", Var("para1"))
                                & Constructor("structure", Var("para2"))
                                & Constructor("structure", Var("para3"))
                                & Constructor("structure", Var("para4"))
                                & Constructor("structure", Var("para5"))
                                & Constructor("structure", Var("para6"))
                                & Constructor("structure", Var("para7"))
                                #& Constructor("structure", Literal("relu"))
                                & Constructor("structure", Literal(None))
                                ) & Constructor("non_ID")
                    ),

            "tanh": SpecificationBuilder()
            .parameter("l", labels, lambda v: list(labels.iter_tanh()))
            .parameter("i", dimension)
            .parameter("o", dimension, lambda v: [v["i"]])
            .parameter("para1", para_labels, lambda v: [(v["l"], v["i"], v["o"])])
            .parameter("para2", para_labels, lambda v: [(v["l"], v["i"], None)])
            .parameter("para3", para_labels, lambda v: [(v["l"], None, v["o"])])
            .parameter("para4", para_labels, lambda v: [(None, v["i"], v["o"])])
            .parameter("para5", para_labels, lambda v: [(v["l"], None, None)])
            .parameter("para6", para_labels, lambda v: [(None, None, v["o"])])
            .parameter("para7", para_labels, lambda v: [(None, v["i"], None)])
            .suffix(Constructor("DAG_component",
                                Constructor("input", Var("i"))
                                & Constructor("input", Literal(None))
                                & Constructor("output", Var("o"))
                                & Constructor("output", Literal(None))
                                & Constructor("structure", Var("para1"))
                                & Constructor("structure", Var("para2"))
                                & Constructor("structure", Var("para3"))
                                & Constructor("structure", Var("para4"))
                                & Constructor("structure", Var("para5"))
                                & Constructor("structure", Var("para6"))
                                & Constructor("structure", Var("para7"))
                                # & Constructor("structure", Literal("relu"))
                                & Constructor("structure", Literal(None))
                                ) & Constructor("non_ID")
                    ),

            "lte": SpecificationBuilder()
            .parameter("l", labels, lambda v: list(labels.iter_lte()))
            .parameter("i", dimension)
            .parameter("o", dimension, lambda v: [v["i"]])
            .parameter("para1", para_labels, lambda v: [(v["l"], v["i"], v["o"])])
            .parameter("para2", para_labels, lambda v: [(v["l"], v["i"], None)])
            .parameter("para3", para_labels, lambda v: [(v["l"], None, v["o"])])
            .parameter("para4", para_labels, lambda v: [(None, v["i"], v["o"])])
            .parameter("para5", para_labels, lambda v: [(v["l"], None, None)])
            .parameter("para6", para_labels, lambda v: [(None, None, v["o"])])
            .parameter("para7", para_labels, lambda v: [(None, v["i"], None)])
            .suffix(Constructor("DAG_component",
                                Constructor("input", Var("i"))
                                & Constructor("input", Literal(None))
                                & Constructor("output", Var("o"))
                                & Constructor("output", Literal(None))
                                & Constructor("structure", Var("para1"))
                                & Constructor("structure", Var("para2"))
                                & Constructor("structure", Var("para3"))
                                & Constructor("structure", Var("para4"))
                                & Constructor("structure", Var("para5"))
                                & Constructor("structure", Var("para6"))
                                & Constructor("structure", Var("para7"))
                                # & Constructor("structure", Literal("relu"))
                                & Constructor("structure", Literal(None))
                                ) & Constructor("non_ID")
                    ),

            "sum": SpecificationBuilder()
            .parameter("l", labels, lambda v: list(labels.iter_sum()))
            .parameter("i", dimension)
            .parameter("o", dimension, lambda v: [1])
            .parameter("para1", para_labels, lambda v: [(v["l"], v["i"], v["o"])])
            .parameter("para2", para_labels, lambda v: [(v["l"], v["i"], None)])
            .parameter("para3", para_labels, lambda v: [(v["l"], None, v["o"])])
            .parameter("para4", para_labels, lambda v: [(None, v["i"], v["o"])])
            .parameter("para5", para_labels, lambda v: [(v["l"], None, None)])
            .parameter("para6", para_labels, lambda v: [(None, None, v["o"])])
            .parameter("para7", para_labels, lambda v: [(None, v["i"], None)])
            .suffix(Constructor("DAG_component",
                                Constructor("input", Var("i"))
                                & Constructor("input", Literal(None))
                                & Constructor("output", Var("o"))
                                & Constructor("output", Literal(None))
                                & Constructor("structure", Var("para1"))
                                & Constructor("structure", Var("para2"))
                                & Constructor("structure", Var("para3"))
                                & Constructor("structure", Var("para4"))
                                & Constructor("structure", Var("para5"))
                                & Constructor("structure", Var("para6"))
                                & Constructor("structure", Var("para7"))
                                #& Constructor("structure", Literal("sum"))
                                & Constructor("structure", Literal(None))
                                ) & Constructor("non_ID")
                    ),

            "product": SpecificationBuilder()
            .parameter("l", labels, lambda v: list(labels.iter_product()))
            .parameter("i", dimension)
            .parameter("o", dimension, lambda v: [1])
            .parameter("para1", para_labels, lambda v: [(v["l"], v["i"], v["o"])])
            .parameter("para2", para_labels, lambda v: [(v["l"], v["i"], None)])
            .parameter("para3", para_labels, lambda v: [(v["l"], None, v["o"])])
            .parameter("para4", para_labels, lambda v: [(None, v["i"], v["o"])])
            .parameter("para5", para_labels, lambda v: [(v["l"], None, None)])
            .parameter("para6", para_labels, lambda v: [(None, None, v["o"])])
            .parameter("para7", para_labels, lambda v: [(None, v["i"], None)])
            .suffix(Constructor("DAG_component",
                                Constructor("input", Var("i"))
                                & Constructor("input", Literal(None))
                                & Constructor("output", Var("o"))
                                & Constructor("output", Literal(None))
                                & Constructor("structure", Var("para1"))
                                & Constructor("structure", Var("para2"))
                                & Constructor("structure", Var("para3"))
                                & Constructor("structure", Var("para4"))
                                & Constructor("structure", Var("para5"))
                                & Constructor("structure", Var("para6"))
                                & Constructor("structure", Var("para7"))
                                #& Constructor("structure", Literal("product"))
                                & Constructor("structure", Literal(None))
                                ) & Constructor("non_ID")
                    ),

            "copy": SpecificationBuilder()
            .parameter("l", labels, lambda v: list(labels.iter_copy()))
            .parameter("i", dimension, lambda v: [1])
            .parameter("o", dimension, lambda v: [v["l"].out_dimension])
            .parameter("para1", para_labels, lambda v: [(v["l"], v["i"], v["o"])])
            .parameter("para2", para_labels, lambda v: [(v["l"], v["i"], None)])
            .parameter("para3", para_labels, lambda v: [(v["l"], None, v["o"])])
            .parameter("para4", para_labels, lambda v: [(None, v["i"], v["o"])])
            .parameter("para5", para_labels, lambda v: [(v["l"], None, None)])
            .parameter("para6", para_labels, lambda v: [(None, None, v["o"])])
            .parameter("para7", para_labels, lambda v: [(None, v["i"], None)])
            .suffix(Constructor("DAG_component",
                                Constructor("input", Var("i"))
                                & Constructor("input", Literal(None))
                                & Constructor("output", Var("o"))
                                & Constructor("output", Literal(None))
                                & Constructor("structure", Var("para1"))
                                & Constructor("structure", Var("para2"))
                                & Constructor("structure", Var("para3"))
                                & Constructor("structure", Var("para4"))
                                & Constructor("structure", Var("para5"))
                                & Constructor("structure", Var("para6"))
                                & Constructor("structure", Var("para7"))
                                # & Constructor("structure", Literal("product"))
                                & Constructor("structure", Literal(None))
                                ) & Constructor("non_ID")
                    ),

            "beside_singleton": SpecificationBuilder()
            .parameter("i", dimension)
            .parameter("o", dimension)
            .parameter("ls", paratuples)
            .parameter_constraint(lambda v: v["ls"] is None or len(v["ls"]) == 1)
            .parameter("para", para_labels, lambda v: [None] if v["ls"] is None else [v["ls"][0]])
            .parameter_constraint(lambda v: (len(v["para"]) == 3 and
                                             (v["para"][1] == v["i"] if v["para"][1] is not None else True) and
                                             (v["para"][2] == v["o"] if v["para"][2] is not None else True)
                                             ) if v["para"] is not None else True)
            .suffix(
                ((Constructor("DAG_component",
                                       Constructor("input", Var("i"))
                                       & Constructor("output", Var("o"))
                                       & Constructor("structure", Var("para")))
                     & Constructor("non_ID")
                  )
                 **
                 (Constructor("DAG_parallel",
                                Constructor("input", Var("i"))
                                & Constructor("input", Literal(None))
                                & Constructor("output", Var("o"))
                                & Constructor("output", Literal(None))
                                & Constructor("structure", Var("ls"))
                                #& Constructor("structure", Literal(None))
                                )
                  & Constructor("non_ID") & Constructor("last", Constructor("non_ID"))
                  )
                 )
                &
                ((Constructor("DAG_component",
                                       Constructor("input", Var("i"))
                                       & Constructor("output", Var("o"))
                                       & Constructor("structure", Var("para")))
                     & Constructor("ID")
                     )
                 **
                 (Constructor("DAG_parallel",
                                Constructor("input", Var("i"))
                                & Constructor("input", Literal(None))
                                & Constructor("output", Var("o"))
                                & Constructor("output", Literal(None))
                                & Constructor("structure", Var("ls"))
                                #& Constructor("structure", Literal(None))
                                )
                     & Constructor("ID")
                     )
                 )),

            "beside_cons": SpecificationBuilder()
            .parameter("i", dimension)
            .parameter("i1", dimension)
            .parameter("i2", dimension, lambda v: [v["i"] - v["i1"]])
            .parameter("o", dimension)
            .parameter("o1", dimension)
            .parameter("o2", dimension, lambda v: [v["o"] - v["o1"]])
            .parameter("ls", paratuples)
            .parameter_constraint(lambda v: v["ls"] is None or len(v["ls"]) > 1)
            .parameter("head", para_labels, lambda v: [None] if v["ls"] is None else [v["ls"][0]])
            .parameter_constraint(lambda v: v["head"] is None or (len(v["head"]) == 3 and
                                                                  (v["head"][1] == v["i1"] or v["head"][1] is None) and
                                                                  (v["head"][2] == v["o1"] or v["head"][2] is None)))
            .parameter("tail", paratuples, lambda v: [None] if v["ls"] is None else [v["ls"][1:]])
            .suffix(
                    ((Constructor("DAG_component",
                                       Constructor("input", Var("i1"))
                                       & Constructor("output", Var("o1"))
                                       & Constructor("structure", Var("head")))
                      & Constructor("ID"))
                    **
                    (Constructor("DAG_parallel",
                                       Constructor("input", Var("i2"))
                                       & Constructor("output", Var("o2"))
                                       & Constructor("structure", Var("tail")))
                     & Constructor("non_ID") & Constructor("last", Constructor("non_ID")))
                     **
                     (Constructor("DAG_parallel",
                                Constructor("input", Var("i"))
                                & Constructor("input", Literal(None))
                                & Constructor("output", Var("o"))
                                & Constructor("output", Literal(None))
                                & Constructor("structure", Var("ls"))
                                #& Constructor("structure", Literal(None))
                                )
                      & Constructor("non_ID") & Constructor("last", Constructor("ID")))
                     )
                    &
                    ((Constructor("DAG_component",
                                  Constructor("input", Var("i1"))
                                  & Constructor("output", Var("o1"))
                                  & Constructor("structure", Var("head")))
                      & Constructor("non_ID"))
                     **
                     (Constructor("DAG_parallel",
                                  Constructor("input", Var("i2"))
                                  & Constructor("output", Var("o2"))
                                  & Constructor("structure", Var("tail")))
                      & Constructor("ID"))
                     **
                     (Constructor("DAG_parallel",
                                  Constructor("input", Var("i"))
                                  & Constructor("input", Literal(None))
                                  & Constructor("output", Var("o"))
                                  & Constructor("output", Literal(None))
                                  & Constructor("structure", Var("ls"))
                                  #& Constructor("structure", Literal(None))
                                  )
                      & Constructor("non_ID") & Constructor("last", Constructor("non_ID")))
                     )
                    &
                    ((Constructor("DAG_component",
                                  Constructor("input", Var("i1"))
                                  & Constructor("output", Var("o1"))
                                  & Constructor("structure", Var("head")))
                      & Constructor("non_ID"))
                     **
                     (Constructor("DAG_parallel",
                                  Constructor("input", Var("i2"))
                                  & Constructor("output", Var("o2"))
                                  & Constructor("structure", Var("tail")))
                      & Constructor("non_ID"))
                     **
                     (Constructor("DAG_parallel",
                                  Constructor("input", Var("i"))
                                  & Constructor("input", Literal(None))
                                  & Constructor("output", Var("o"))
                                  & Constructor("output", Literal(None))
                                  & Constructor("structure", Var("ls"))
                                  #& Constructor("structure", Literal(None))
                                  )
                      & Constructor("non_ID") & Constructor("last", Constructor("non_ID")))
                     )
                    ),

            # normalization is already done at learner combinator and may be removed here, but this would require refactoring of term predicates...
            "before_singleton": SpecificationBuilder()
            .parameter("i", dimension)
            .parameter("o", dimension)
            .parameter("request", paratupletuples)
            .parameter("ls", paratupletuples, lambda v: [paratupletuples.normalize(v["request"])])
            .parameter_constraint(lambda v: v["ls"] is not None and len(v["ls"]) == 1)
            .parameter("ls1", paratuples, lambda v: [v["ls"][0]])
            .parameter_constraint(lambda v: v["ls1"] is None or
                                            (
                                                    (
                                                        v["i"] == sum([t[1] for t in v["ls1"]])
                                                        if None not in [t for t in v["ls1"]]
                                                           and None not in [t[1] for t in v["ls1"]]
                                                        else v["i"] > sum([t[1] for t in v["ls1"]
                                                                           if t is not None
                                                                           and t[1] is not None])
                                                    )
                                                    and
                                                    (
                                                        v["o"] == sum([t[2] for t in v["ls1"]])
                                                        if None not in [t for t in v["ls1"]]
                                                           and None not in [t[2] for t in v["ls1"]]
                                                        else v["o"] > sum([t[2] for t in v["ls1"]
                                                                           if t is not None
                                                                           and t[2] is not None]))
                                            )
                                  )
            .argument("x", Constructor("DAG_parallel",
                                       Constructor("input", Var("i"))
                                       & Constructor("output", Var("o"))
                                       & Constructor("structure", Var("ls1"))) & Constructor("non_ID"))
            .suffix(Constructor("DAG",
                                Constructor("input", Var("i"))
                                & Constructor("input", Literal(None))
                                & Constructor("output", Var("o"))
                                & Constructor("output", Literal(None))
                                & Constructor("structure", Var("request"))
                                )),

            # normalization is already done at learner combinator and may be removed here, but this would require refactoring of term predicates...
            "before_cons": SpecificationBuilder()
            .parameter("i", dimension)
            .parameter("j", dimension)
            .parameter("o", dimension)
            .parameter("request", paratupletuples)
            .parameter("ls", paratupletuples, lambda v: [paratupletuples.normalize(v["request"])])
            .parameter_constraint(lambda v: v["ls"] is not None and len(v["ls"]) > 1)
            .parameter("head", paratuples, lambda v: [v["ls"][0]])
            .parameter_constraint(lambda v: v["head"] is None or
                                            (
                                                (v["i"] == sum([t[1] for t in v["head"]])
                                                 if None not in [t for t in v["head"]]
                                                    and None not in [t[1] for t in v["head"]]
                                                 else v["i"] > sum([t[1] for t in v["head"]
                                                                    if t is not None and t[1] is not None]))
                                                and (v["j"] == sum([t[2] for t in v["head"]])
                                                     if None not in [t for t in v["head"]]
                                                        and None not in [t[2] for t in v["head"]]
                                                     else v["j"] > sum([t[2] for t in v["head"]
                                                                        if t is not None and t[2] is not None]))
                                            )
                                  )
            .parameter("tail", paratupletuples, lambda v: [v["ls"][1:]])
            .parameter_constraint(lambda v: v["tail"] is None or
                                            (
                                                    (len(v["tail"]) > 0) and
                                                    (
                                                            v["tail"][0] is None or
                                                            (
                                                                v["j"] == sum([t[1] for t in v["tail"][0]])
                                                                if None not in [t for t in v["tail"][0]]
                                                                   and None not in [t[1] for t in v["tail"][0]]
                                                                else v["j"] > sum([t[1] for t in v["tail"][0]
                                                                                   if t is not None
                                                                                   and t[1] is not None])
                                                             )
                                                    ) and
                                                    (
                                                            v["tail"][-1] is None or
                                                            (v["o"] == sum([t[2] for t in v["tail"][-1]])
                                                             if None not in [t for t in v["tail"][-1]]
                                                                and None not in [t[2] for t in v["tail"][-1]]
                                                             else v["o"] > sum([t[2] for t in v["tail"][-1]
                                                                                if t is not None
                                                                                and t[2] is not None]))
                                                    )
                                            )
                                  )
            .argument("x", Constructor("DAG_parallel",
                                       Constructor("input", Var("i"))
                                       & Constructor("output", Var("j"))
                                       & Constructor("structure", Var("head"))) & Constructor("non_ID"))
            .argument("y", Constructor("DAG",
                                       Constructor("input", Var("j"))
                                       & Constructor("output", Var("o"))
                                       & Constructor("structure", Var("tail"))))
            .constraint(lambda v: self.swaplaw1(v["x"], v["y"]))
            .constraint(lambda v: self.swaplaw2(v["x"], v["y"]))
            .constraint(lambda v: self.swaplaw3(v["x"], v["y"]))
            .constraint(lambda v: self.swaplaw4(v["x"], v["y"]))
            .suffix(Constructor("DAG",
                                Constructor("input", Var("i"))
                                & Constructor("input", Literal(None))
                                & Constructor("output", Var("o"))
                                & Constructor("output", Literal(None))
                                & Constructor("structure", Var("request")))),

            "mse_loss": SpecificationBuilder()
            .parameter("loss", loss_function, lambda v: list(loss_function.iter_mseloss()))
            .suffix(Constructor("Loss", Constructor("type", Var("loss")) & Constructor("type", Literal(None)))),

            "l1loss": SpecificationBuilder()
            .parameter("loss", loss_function, lambda v: list(loss_function.iter_l1loss()))
            .suffix(Constructor("Loss", Constructor("type", Var("loss")) & Constructor("type", Literal(None)))),

            "adam_optimizer": SpecificationBuilder()
            .parameter("optimizer", optimizer)
            .suffix(Constructor("Optimizer", Constructor("type", Var("optimizer")) & Constructor("type", Literal(None)))),

            "learner": SpecificationBuilder()
            .parameter("i", dimension_with_None)
            .parameter("o", dimension_with_None)
            .parameter("request", paratupletuples)
            .parameter("ls", paratupletuples, lambda v: [paratupletuples.normalize(v["request"])])
            .parameter("epochs", epochs)
            .parameter("loss", loss_function)
            .parameter("opti", optimizer)
            .argument("loss_f", Constructor("Loss", Constructor("type", Var("loss"))))
            .argument("optimizer", Constructor("Optimizer", Constructor("type", Var("opti"))))
            .argument("model", Constructor("DAG",
                                           Constructor("input", Var("i"))
                                           & Constructor("output", Var("o"))
                                           & Constructor("structure", Var("ls"))))
            .suffix(Constructor("Learner", Constructor("DAG",
                                                       Constructor("input", Var("i"))
                                                       & Constructor("output", Var("o"))
                                                       & Constructor("structure", Var("request"))
                                                       )
                                & Constructor("Loss", Constructor("type", Var("loss")))
                                & Constructor("Optimizer", Constructor("type", Var("opti")))
                                & Constructor("epochs", Var("epochs"))
                                )
                    )
        }

    # Interpretations of terms are algebras in my language

    def pretty_term_algebra(self):
        return {
            "edges": (lambda io, para1, para2, para3, para4, para5, para6, para7, para8, para9, para10, para11, para12, para13, para14, para15, para16: f"edges({io})"),

            "swap": (lambda io, n, m, para1, para2, para3, para4, para5, para6, para7, para8, para9, para10, para11, para12, para13, para14, para15, para16: f"swap({io}, {n}, {m})"),

            "linear_layer": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: f"({str(l)}, {i}, {o})"),

            "sigmoid": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: f"({str(l)}, {i}, {o})"),

            "relu": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: f"({str(l)}, {i}, {o})"),

            "tanh": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: f"({str(l)}, {i}, {o})"),

            "lte": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: f"({str(l)}, {i}, {o})"),

            "sum": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: f"({str(l)}, {i}, {o})"),

            "product": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: f"({str(l)}, {i}, {o})"),

            "copy": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: f"({str(l)}, {i}, {o})"),

            "beside_singleton": (lambda i, o, ls, para, x: f"{x})"),

            "beside_cons": (lambda i, i1, i2, o, o1, o2, ls, head, tail, x, y: f"{x} || {y}"),

            "before_singleton": (lambda i, o, r, ls, ls1, x: f"({x}"),

            "before_cons": (lambda i, j, o, r, ls, head, tail, x, y: f"({x} ; {y}"),

            "mse_loss": (lambda l: str(l)),

            "l1loss": (lambda l: str(l)),

            "adam_optimizer": (lambda o: str(o)),

            "learner": (lambda i, o, r, ls, e, l, opt, loss, optimizer, model: f"""
Learner(
    model= (
        {model}
        ), 
    loss= {loss}, 
    optimizer= {optimizer}, 
    epochs= {e}
    )
""")
        }

    def edgelist_learner(self, model, loss, optimizer, epochs):
        f, inputs = model
        edgelist, to_outputs, pos_A = f((-3.8, -3.8), ["input" for _ in range(0, inputs)])
        edgelist = edgelist + [(o, loss) for o in to_outputs] + [(loss, optimizer)] + [(optimizer, f"epochs({epochs})")]
        return edgelist


    def edgelist_algebra(self):
        return {
            "edges": (lambda io, para1, para2, para3, para4, para5, para6, para7, para8, para9, para10, para11, para12, para13, para14, para15, para16: lambda id, inputs: ([], inputs, {})),

            "swap": (lambda io, n, m, para1, para2, para3, para4, para5, para6, para7, para8, para9, para10, para11, para12, para13, para14, para15, para16: lambda id, inputs: ([], inputs[n:] + inputs[:n], {})),

            "linear_layer": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: lambda id, inputs: (
                [(x, str((l, i, o)) + str(id)) for x in inputs], [str((l, i, o)) + str(id) for _ in range(0, o)],
                {str((l, i, o)) + str(id): id})),

            "sigmoid": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: lambda id, inputs: (
                [(x, str((l, i, o)) + str(id)) for x in inputs], [str((l, i, o)) + str(id) for _ in range(0, o)],
                {str((l, i, o)) + str(id): id})),

            "relu": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: lambda id, inputs: (
                [(x, str((l, i, o)) + str(id)) for x in inputs], [str((l, i, o)) + str(id) for _ in range(0, o)],
                {str((l, i, o)) + str(id): id})),

            "tanh": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: lambda id, inputs: (
                [(x, str((l, i, o)) + str(id)) for x in inputs], [str((l, i, o)) + str(id) for _ in range(0, o)],
                {str((l, i, o)) + str(id): id})),

            "lte": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: lambda id, inputs: (
                [(x, str((l, i, o)) + str(id)) for x in inputs], [str((l, i, o)) + str(id) for _ in range(0, o)],
                {str((l, i, o)) + str(id): id})),

            "sum": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: lambda id, inputs: (
                [(x, str((l, i, o)) + str(id)) for x in inputs], [str((l, i, o)) + str(id) for _ in range(0, o)],
                {str((l, i, o)) + str(id): id})),

            "product": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: lambda id, inputs: (
                [(x, str((l, i, o)) + str(id)) for x in inputs], [str((l, i, o)) + str(id) for _ in range(0, o)],
                {str((l, i, o)) + str(id): id})),

            "copy": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: lambda id, inputs: (
                [(x, str((l, i, o)) + str(id)) for x in inputs], [str((l, i, o)) + str(id) for _ in range(0, o)],
                {str((l, i, o)) + str(id): id})),

            "beside_singleton": (lambda i, o, ls, para, x: x),

            "beside_cons": (lambda i, i1, i2, o, o1, o2, ls, head, tail, x, y: lambda id, inputs:
                (x(id, inputs[:i1])[0] + y((id[0], id[1] + 0.2), inputs[i1:])[0],
                 x(id, inputs[:i1])[1] + y((id[0], id[1] + 0.2), inputs[i1:])[1],
                 x(id, inputs[:i1])[2] | y((id[0], id[1] + 0.2), inputs[i1:])[2])),

            "before_singleton": (lambda i, o, r, ls, ls1, x: (x, i)),

            "before_cons": (lambda i, j, o, r, ls, head, tail, x, y: (lambda id, inputs:
                                                                      (
                                                                           y[0]((id[0] + 2.5, id[1]), x(id, inputs)[1])[0] + x(id, inputs)[0],
                                                                           y[0]((id[0] + 2.5, id[1]), x(id, inputs)[1])[1],
                                                                           y[0]((id[0] + 2.5, id[1]), x(id, inputs)[1])[2] | x(id, inputs)[2]
                                                                       ),
                                                                       i)),

            "mse_loss": (lambda l: str(l)),

            "l1loss": (lambda l: str(l)),

            "adam_optimizer": (lambda o: str(o)),

            "learner": (lambda i, o, r, ls, e, l, opt, loss, optimizer, model: self.edgelist_learner(model, loss, optimizer, e))
        }

    def pytorch_code_algebra(self):
        # construct the __init__ part and forward function in parallel
        # register a module in __init__, give it a unique name (thats why we abstract over an id), and use that name in forward
        # should be similar to edgelist algebra
        # assume, that x is a vector with the length of the input dimension
        # we need to return a triple of two strings and the output vector (then its really similar to edgelist, yippieh!!!)
        # TODO: update pytorch code algebra to implementation of pytorch function algebra
        return {
            "edges": (lambda io, para1, para2, para3, para4, para5, para6, para7, para8, para9, para10, para11, para12, para13, para14, para15, para16: lambda id, inputs: (f"""""", f"""""", inputs)),

            "swap": (lambda io, n, m, para1, para2, para3, para4, para5, para6, para7, para8, para9, para10, para11, para12, para13, para14, para15, para16: lambda id, inputs: (f"""""", f"""""", inputs[n:] + inputs[:n])),

            "linear_layer": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: lambda id, inputs: (f"""
        self.linear_{id[0]}_{id[1]} = nn.Linear({l.in_features}, {l.out_features}, {l.bias}) 
""", "\n".join([f"""
        x_{id[0]}_{id[1]}_{k} = self.linear_{id[0]}_{id[1]}({", ".join(inputs)})
""" for k in range(o)]), tuple(f"x_{id[0]}_{id[1]}_{k}" for k in range(o)))),

            "sigmoid": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: lambda id, inputs: (f"""
        self.sigmoid_{id[0]}_{id[1]} = nn.Sigmoid()
""", "\n".join([f"""
        x_{id[0]}_{id[1]}_{k} = self.sigmoid_{id[0]}_{id[1]}({", ".join(inputs)})
""" for k in range(o)]), tuple(f"x_{id[0]}_{id[1]}_{k}" for k in range(o)))),

            "relu": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: lambda id, inputs: (f"""
        self.relu_{id[0]}_{id[1]} = nn.ReLU(inplace={l.inplace})
""", "\n".join([f"""
        x_{id[0]}_{id[1]}_{k} = self.relu_{id[0]}_{id[1]}({", ".join(inputs)})
""" for k in range(o)]), tuple(f"x_{id[0]}_{id[1]}_{k}" for k in range(o)))),

            "tanh": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: lambda id, inputs: (f"""
                self.tanh_{id[0]}_{id[1]} = nn.Tanh()
        """, "\n".join([f"""
                x_{id[0]}_{id[1]}_{k} = self.tanh_{id[0]}_{id[1]}({", ".join(inputs)})
        """ for k in range(o)]), tuple(f"x_{id[0]}_{id[1]}_{k}" for k in range(o)))),

            "lte": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: lambda id, inputs: (f"""
                        self.lte_{id[0]}_{id[1]} = ?
                """, "\n".join([f"""
                        x_{id[0]}_{id[1]}_{k} = self.tanh_{id[0]}_{id[1]}({", ".join(inputs)})
                """ for k in range(o)]), tuple(f"x_{id[0]}_{id[1]}_{k}" for k in range(o)))),

            "sum": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: lambda id, inputs: (f"""""","\n".join([f"""
        x_{id[0]}_{id[1]}_{k} = {l.with_constant} + {" + ".join(inputs)}
""" for k in range(o)]), tuple(f"x_{id[0]}_{id[1]}_{k}" for k in range(o)))),

            "product": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: lambda id, inputs: (f"""""","\n".join([f"""
        x_{id[0]}_{id[1]}_{k} = {l.with_constant} * {" * ".join(inputs)}
""" for k in range(o)]), tuple(f"x_{id[0]}_{id[1]}_{k}" for k in range(o)))),

            "copy": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: lambda id, inputs: (f"""""",
                                                                                                             "\n".join(
                                                                                                                 [f"""
                x_{id[0]}_{id[1]}_{k} = {l.with_constant} * {" * ".join(inputs)}
        """ for k in range(o)]), tuple(f"x_{id[0]}_{id[1]}_{k}" for k in range(o)))),

            "beside_singleton": (lambda i, o, ls, para, x: x),

            "beside_cons": (lambda i, i1, i2, o, o1, o2, ls, head, tail, x, y: lambda id, inputs: (
                x(id, inputs[:i1])[0] + y((id[0], id[1] + 1), inputs[i1:])[0], # concatenate inits
                x(id, inputs[:i1])[1] + y((id[0], id[1] + 1), inputs[i1:])[1], # concatenate forwards
                x(id, inputs[:i1])[2] + y((id[0], id[1] + 1), inputs[i1:])[2])), # concatenate output vectors

            "before_singleton": (lambda i, o, r, ls, ls1, x: x),

            "before_cons": (lambda i, j, o, r, ls, head, tail, x, y: lambda id, inputs: (
                x((id[0], id[1]), inputs)[0] + y((id[0] + 1, id[1]), x((id[0], id[1]), inputs)[2])[0], # concatenate inits
                x((id[0], id[1]), inputs)[1] + y((id[0] + 1, id[1]), x((id[0], id[1]), inputs)[2])[1], # concatenate forwards
                y((id[0] + 1, id[1]), x((id[0], id[1]), inputs)[2])[2])), # output vector is the output of y

            "mse_loss": (lambda l: f"nn.MSELoss(reduction='{l.reduction}')"),

            "l1loss": (lambda l: f"nn.MSELoss(reduction='{l.reduction}')"),

            "adam_optimizer": (lambda o: lambda m: f"optim.Adam({m}.parameters(), lr={o.learning_rate})"),

            "learner": (lambda i, o, r, ls, e, l, opt, loss, optimizer, model: f"""
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import tqdm

def fit_model(model, x, y, n_epochs=2_000, verbose=True):
    optimizer = {optimizer("model")}
    loss_fn = {loss}
    
    pbar = tqdm.tqdm(range(n_epochs), total=n_epochs, desc=f"Training SynthesizedModel", disable=not verbose)
    
    for _ in pbar:
        optimizer.zero_grad()
        pred = model(x).ravel()
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        
        pbar.set_postfix({"{"}"loss": f"{"{"}loss.item():.6f{"}"}"{"}"})
    
    return model

class SynthesizedModel(nn.Module):
    def __init__(self)
{model((0,0), tuple(("x" for _ in range(i))))[0]}

    def forward(self, x):
{model((0,0), tuple(("x" for _ in range(i))))[1]}
        return {', '.join(model((0,0), tuple(("x" for _ in range(i))))[2])}
        
# Andreas will provide a dataloader for training data and test data
x,y = ANDREAS!!!!!!!!!
x_test,y_test = ANDREAS!!!!!!!!!

m = SynthesizedModel()
trained_model = fit_model(m, x, y, n_epochs={e})

loss_fn = {loss}
with torch.inference_mode():
    y_pred = trained_model(x_test).ravel()
    loss = loss_fn(y_pred, y_test)
    print("Test Loss: " + loss.item())
    
# Plot true function
plt.figure(figsize=(12, 8))
plt.plot(x_test.view(-1), y_test, label="True Trapezoid", linewidth=3)

with torch.inference_mode():
    y_pred = trained_model(x_test).ravel()
    plt.plot(x_test.view(-1), y_pred, label=f"Learned SynthesizedModel", linestyle="--")
    
plt.title("True vs Learned Trapezoid Mapping")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.savefig("plot.pdf")
plt.savefig("plot.png")
plt.show()
""")
        }

    class EdgesModule(nn.Module):
        # nn.Identity? our forward is tuple of tensors -> tuple of tensors, so maybe not...
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return x

    class SwapModule(nn.Module):
        def __init__(self, n: int, m: int):
            super().__init__()
            self.n = n
            self.m = m

        def forward(self, x):
            x1, x2 = torch.split(x, [self.n, len(x[1]) - self.n], dim=1)
            if len(x1[1]) != self.n or len(x2[1]) != self.m:
                raise ValueError(f"Swap Module expected input dimensions ({self.n}, {self.m}), but got ({len(x1[1])}, {len(x2[1])})")
            return torch.cat((x2, x1), dim=1)

    class SynthLinear(nn.Module):
        def __init__(self, output_dim, in_features, out_features, bias=True):
            super().__init__()
            self.linear = nn.Linear(in_features, out_features, bias)
            self.o = output_dim
            self.in_features = in_features

        def forward(self, x):
            #if len(x) == 1:
            #    x = x[0]
            #else:
             #   x = torch.cat(x, dim=1)
            if len(x[1]) != self.in_features:
                raise ValueError(f"Linear Layer expected {self.in_features} inputs, but got {len(x)}")
            y = self.linear(x)
            if len(y[1]) != self.o:
                raise ValueError(f"Linear Layer expected to produce {self.o} outputs, but got {len(y)}")
            return y

    class SynthSigmoid(nn.Module):
        def __init__(self, output_dim):
            super().__init__()
            self.sigmoid = nn.Sigmoid()
            self.o = output_dim

        def forward(self, x):
            #x = torch.cat(x, dim=1)
            if len(x[1]) != self.o:
                raise ValueError(f"Sigmoid expected {self.o} inputs, but got {len(x)}")
            y = self.sigmoid(x)
            if len(y[1]) != self.o:
                raise ValueError(f"Sigmoid expected to produce {self.o} outputs, but got {len(y)}")
            return y

    class SynthReLU(nn.Module):
        def __init__(self, output_dim, inplace=False):
            super().__init__()
            self.relu = nn.ReLU(inplace=inplace)
            self.o = output_dim

        def forward(self, x):
            #x = torch.cat(x, dim=1)
            if len(x[1]) != self.o:
                raise ValueError(f"ReLU expected {self.o} inputs, but got {len(x)}")
            y = self.relu(x)
            if len(y[1]) != self.o:
                raise ValueError(f"ReLU expected to produce {self.o} outputs, but got {len(y)}")
            return y

    class SynthTanh(nn.Module):
        def __init__(self, output_dim):
            super().__init__()
            self.tanh = nn.Tanh()
            self.o = output_dim

        def forward(self, x):
            #x = torch.cat(x, dim=1)
            if len(x[1]) != self.o:
                raise ValueError(f"Sigmoid expected {self.o} inputs, but got {len(x)}")
            y = self.tanh(x)
            if len(y[1]) != self.o:
                raise ValueError(f"Sigmoid expected to produce {self.o} outputs, but got {len(y)}")
            return y

    class SynthLTE(nn.Module):
        def __init__(self, output_dim, threshold:float=0.0):
            super().__init__()
            self.threshold = threshold
            self.o = output_dim

        def forward(self, x):
            #x = torch.cat(x, dim=1)
            if len(x[1]) != self.o:
                raise ValueError(f"LTE expected {self.o} inputs, but got {len(x)}")
            y = (x <= self.threshold).float()
            if len(y[1]) != self.o:
                raise ValueError(f"LTE expected to produce {self.o} outputs, but got {len(y)}")
            return y

    class SumModule(nn.Module):
        def __init__(self, output_dim, with_constant):
            super().__init__()
            self.with_constant = with_constant
            self.o = output_dim

        def forward(self, x):
            #x = torch.cat(x, dim=1)
            x = torch.add(x, self.with_constant)
            x = torch.sum(x, dim=1, keepdim=True)
            return x

    class ProductModule(nn.Module):
        def __init__(self, output_dim, with_constant):
            super().__init__()
            self.with_constant = with_constant
            self.o = output_dim

        def forward(self, x):
            #x = torch.cat(x, dim=1)
            x = torch.mul(x, self.with_constant)
            x = torch.prod(x, dim=1, keepdim=True)
            return x

    class BesideModule(nn.Module):
        def __init__(self, head, tail, i1: int):
            super().__init__()
            self.head = head
            self.tail = tail
            self.i = i1

        def forward(self, x):
            if self.i == len(x[0]):
                if self.tail is not None:
                    raise ValueError("BesideModule: tail is not None, but input dimension matches head dimension")
                return self.head(x)
            else:
                x1, x2 = torch.split(x, [self.i, len(x[1]) - self.i], dim=1)
                head_out = self.head(x1)
                tail_out = self.tail(x2)
                output = torch.cat((head_out, tail_out), dim=1)
                return output

    class BeforeModule(nn.Module):
        # nn.Sequential basically, but the type of our forward is tuple of tensors -> tuple of tensors, I want to make sure nothing unforeseen happens
        def __init__(self, head, tail):
            super().__init__()
            self.head = head
            self.tail = tail if tail is not None else lambda x: x  # catch beside singleton

        def forward(self, x):
            #x = torch.cat(x, dim=1)
            head_out = self.head(x)
            tail_out = self.tail(head_out)
            return tail_out

    class CopyModule(nn.Module):
        def __init__(self, output_dim):
            super().__init__()
            self.o = output_dim

        def forward(self, x):
            return x.expand(len(x), self.o).clone()

    def learner(self, i, open_model, loss_fn, optim, n_epochs, x, y, x_test, y_test):
        # training loop for the synthesized model

        model = open_model
        parameter_list = list(model.parameters())
        if len(parameter_list) > 0:

            # fit model
            optimizer = optim(model)
            for _ in range(n_epochs):
                optimizer.zero_grad()
                pred = model(x).ravel()
                loss = loss_fn(pred, y)
                loss.backward()
                optimizer.step()

        with torch.inference_mode():
            y_pred = model(x_test).ravel()
            loss = loss_fn(y_pred, y_test)
            return loss.item()

    def pytorch_function_algebra(self):
        # Use nn.ModuleDict to store layers with unique names (that's why we abstract over an id) and implement it as the pytorch algebra
        # nn.ModuleDict then becomes the __init__ part and the forward function is constructed accordingly
        return {
            "edges": (lambda io, para1, para2, para3, para4, para5, para6, para7, para8, para9, para10, para11, para12, para13, para14, para15, para16: self.EdgesModule()),

            "swap": (lambda io, n, m, para1, para2, para3, para4, para5, para6, para7, para8, para9, para10, para11, para12, para13, para14, para15, para16: self.SwapModule(n, m)),

            "linear_layer": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: self.SynthLinear(o, l.in_features, l.out_features, l.bias)),

            "sigmoid": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: self.SynthSigmoid(o)),

            "relu": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: self.SynthReLU(o, l.inplace)),

            "tanh": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: self.SynthTanh(o)),

            "lte": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: self.SynthLTE(o, l.threshold)),

            "sum": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: self.SumModule(o, l.with_constant)),

            "product": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: self.ProductModule(o, l.with_constant)),

            "copy": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: self.CopyModule(o)),

            "beside_singleton": (lambda i, o, ls, para, x: self.BesideModule(x, None, i)),

            "beside_cons": (lambda i, i1, i2, o, o1, o2, ls, head, tail, x, y: self.BesideModule(x, y, i1)),

            "before_singleton": (lambda i, o, r, ls, ls1, x: self.BeforeModule(x, None)),

            "before_cons": (lambda i, j, o, r, ls, head, tail, x, y: self.BeforeModule(x, y)),

            "mse_loss": (lambda l: nn.MSELoss(reduction=l.reduction)),

            "l1loss": (lambda l: nn.L1Loss(reduction=l.reduction)),

            "adam_optimizer": (lambda o: lambda m: optim.Adam(m.parameters(), lr=o.learning_rate)),

            "learner": (lambda i, o, r, ls, e, l, opt, loss, optimizer, model: lambda x, y, x_test, y_test: self.learner(i, model, loss, optimizer, e, x, y, x_test, y_test)),
        }



if __name__ == "__main__":
    repo = ODE_2_Repository(linear_feature_dimensions=[1, 2, 5], constant_values=[0, 1, (-1)],
                            learning_rate_values=[1e-2], n_epoch_values=[10000])


    edge = (("swap", 0, 1), 1, 1)

    def parallel_edges(n):
        if n <= 0:
            raise ValueError("n must be positive")
        else:
            return (("swap", 0, n), n, n)

    target1 = Constructor("DAG",
                          Constructor("input", Literal(1))
                          & Constructor("output", Literal(1))
                          & Constructor("structure", Literal(
                              ((None,), (None,), (None,), (None,),)
                          )))

    target2 = Constructor("Learner", target1
                                & Constructor("Loss", Constructor("type", Literal(None)))
                                & Constructor("Optimizer", Constructor("type", Literal(repo.Adam(1e-2))))
                                & Constructor("epochs", Literal(10000))
                                )


    target_empty = Constructor("Learner", Constructor("DAG",
                          Constructor("input", Literal(1))
                          & Constructor("output", Literal(1))
                          & Constructor("structure", Literal(
                              (
                                  (
                                      (repo.Linear(1, 5, True), 1, 1),
                                  ),
                                  (
                                      (repo.Linear(1, 1, True), 1, 1),
                                  ),
                              )
                          )))
                                & Constructor("Loss", Constructor("type", Literal(None)))
                                & Constructor("Optimizer", Constructor("type", Literal(repo.Adam(1e-2))))
                                & Constructor("epochs", Literal(10000))
                                )

    target_max_seq_3 = Constructor("Learner", Constructor("DAG",
                          Constructor("input", Literal(1))
                          & Constructor("output", Literal(1))
                          & Constructor("structure", Literal(
                              (None, None, None)
                          )))
                                & Constructor("Loss", Constructor("type", Literal(repo.MSEloss())))
                                & Constructor("Optimizer", Constructor("type", Literal(repo.Adam(1e-2))))
                                & Constructor("epochs", Literal(10000))
                                )

    target_trapezoid = Constructor("Learner", Constructor("DAG",
                                                          Constructor("input", Literal(1))
                                                          & Constructor("output", Literal(1))
                                                          & Constructor("structure", Literal(
                                                              (
                                                                  (
                                                                      (ODE_2_Repository.Copy(3), 1, 3),
                                                                  ),
                                                                  (
                                                                      (repo.Linear(1, 1, True), 1, 1),
                                                                      (repo.Linear(1, 1, True), 1, 1),
                                                                      (repo.Linear(1, 1, True), 1, 1)
                                                                  ),  # left, split, right
                                                                  (
                                                                      edge,
                                                                      (repo.LTE(0), 1, 1),
                                                                      edge
                                                                  ),  # left, gate, right
                                                                  (
                                                                      edge,
                                                                      (repo.Copy(2), 1, 2),
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
                                                                            (repo.LTE(0), 1, 2),
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

    target = target_trapezoid
    synthesizer = SearchSpaceSynthesizer(repo.specification(), {})

    search_space = synthesizer.construct_search_space(target).prune()
    print("finish synthesis, start sampling")

    terms =  search_space.enumerate_trees(target, 100)

    #terms = search_space.sample(10, target)

    terms_list = list(terms)

    print("sampling finished")

    #term = terms_list[0]

    print(target)

    print(f"number of terms: {len(terms_list)}")
    """
    for t in terms_list:
        print(t.interpret(repo.pretty_term_algebra()))
    """

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

    for t in terms_list:
        print(t.interpret(repo.pretty_term_algebra()))
        learner = t.interpret(repo.pytorch_function_algebra())
        print("Test Loss: " + str(learner(x, y, x_test, y_test)))
