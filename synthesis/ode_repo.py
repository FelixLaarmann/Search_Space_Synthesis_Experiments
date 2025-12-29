from cl3s import (SpecificationBuilder, Constructor, Literal, Var, Group, DataGroup,
                  DerivationTree)
from typing import Any

from dataclasses import dataclass

class ODE_DAG_Repository:
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

    def __init__(self, dimensions, linear_feature_dimensions, sharpness_values, threshold_values, learning_rate_values,
                 adam_learning_rate_values, n_epoch_values):
        # additionally to labeled nodes, we have (unlabelled) edges, that needs to be handled additionally
        self.dimensions = dimensions
        self.linear_feature_dimensions = linear_feature_dimensions
        self.sharpness_values = sharpness_values
        self.threshold_values = threshold_values
        self.learning_rate_values = learning_rate_values
        self.adam_learning_rate_values = adam_learning_rate_values
        self.n_epoch_values = n_epoch_values
        self.id_seed = 0

    @dataclass(frozen=True)
    class Linear:
        in_features: int | None
        out_features: int | None
        bias: bool | None = True

    @dataclass(frozen=True)
    class Sigmoid:
        trivial = ()

    @dataclass(frozen=True)
    class ReLu:
        inplace: bool | None = False

    @dataclass(frozen=True)
    class Sharpness_Sigmoid:
        sharpness: float | None

    @dataclass(frozen=True)
    class lte:
        threshold: float | None

    @dataclass(frozen=True)
    class Join:
        trivial = ()

    @dataclass(frozen=True)
    class Swap:
        swap_upper_n: int | None
        with_lower_m: int | None

    class Label(Group):
        name = "Label"

        def __init__(self, dimensions, linear_feature_dimensions, sharpness_values, threshold_values):
            self.dimensions = dimensions
            self.linear_feature_dimensions = linear_feature_dimensions
            self.sharpness_values = sharpness_values
            self.threshold_values = threshold_values

        def iter_linear(self):
            for in_f in self.linear_feature_dimensions:
                for out_f in self.linear_feature_dimensions:
                    yield ODE_DAG_Repository.Linear(in_features=in_f, out_features=out_f, bias=True)
                    yield ODE_DAG_Repository.Linear(in_features=in_f, out_features=out_f, bias=False)

        def iter_sigmoid(self):
            yield ODE_DAG_Repository.Sigmoid()

        def iter_relu(self):
            yield ODE_DAG_Repository.ReLu(inplace=False)
            yield ODE_DAG_Repository.ReLu(inplace=True)

        def iter_sharpness_sigmoid(self):
            for sharpness in self.sharpness_values:
                yield ODE_DAG_Repository.Sharpness_Sigmoid(sharpness=sharpness)

        def iter_lte(self):
            for threshold in self.threshold_values:
                yield ODE_DAG_Repository.lte(threshold=threshold)

        def iter_id(self):
            yield ODE_DAG_Repository.Join()

        def iter_swap(self):
            for n in self.dimensions:
                for m in self.dimensions:
                    yield ODE_DAG_Repository.Swap(swap_upper_n=n, with_lower_m=m)

        def __iter__(self):
            yield from self.iter_linear()
            yield from self.iter_sigmoid()
            yield from self.iter_relu()
            yield from self.iter_sharpness_sigmoid()
            yield from self.iter_lte()
            yield from self.iter_id()
            #yield from self.iter_swap()  # swap is not a node label!

        def __contains__(self, item):
            if item is None:
                return True
            elif isinstance(item, ODE_DAG_Repository.Linear):
                return ((item.in_features is None or item.in_features in self.linear_feature_dimensions) and
                        (item.out_features is None or item.out_features in self.linear_feature_dimensions) and
                        (item.bias is None or isinstance(item.bias, bool)))
            elif isinstance(item, ODE_DAG_Repository.Sigmoid):
                return True
            elif isinstance(item, ODE_DAG_Repository.ReLu):
                return item.inplace is None or isinstance(item.inplace, bool)
            elif isinstance(item, ODE_DAG_Repository.Sharpness_Sigmoid):
                return item.sharpness is None or item.sharpness in self.sharpness_values
            elif isinstance(item, ODE_DAG_Repository.lte):
                return item.threshold is None or item.threshold in self.threshold_values
            elif isinstance(item, ODE_DAG_Repository.Join):
                return True
            #elif isinstance(item, ODE_DAG_Repository.Swap):
            #    return ((item.swap_upper_n or None or item.swap_upper_n in self.dimensions) and
            #            (item.with_lower_m is None or item.with_lower_m in self.dimensions))
            # swap is not a node label!
            else:
                return False

    class Para(Group):
        name = "Para"

        def __init__(self, labels, dimensions):
            self.labels = list(labels) + [None]
            self.dimensions = list(dimensions) + [None]

        def __iter__(self):
            for i in self.dimensions:
                if i is not None:
                    for o in self.dimensions:
                        if o is not None:
                            for l in self.labels:
                                if l is not None:
                                    yield l, i, o
                            if i == o:
                                for n in range(0, i):
                                    m = i - n
                                    assert m > 0
                                    yield ODE_DAG_Repository.Swap(n, m), i, o
            yield None


        def __contains__(self, value):
            return value is None or (isinstance(value, tuple)
                                     and len(value) == 3
                                     and (value[0] in self.labels or isinstance(value[0], ODE_DAG_Repository.Swap))
                                     and value[1] in self.dimensions
                                     and value[2] in self.dimensions
                                     and ((value[1] == value[2]
                                          and (value[0].swap_upper_n or None or value[0].swap_upper_n in self.dimensions)
                                          and (value[0].with_lower_m is None or value[0].with_lower_m in self.dimensions))
                                          if isinstance(value[0], ODE_DAG_Repository.Swap) else True))

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
                        if (isinstance(label, ODE_DAG_Repository.Swap) and label.swap_upper_n == 0):
                            if len(r) == 3 and r[0] is not None:
                                right_label, right_i, right_o = r
                                if (isinstance(right_label, ODE_DAG_Repository.Swap)
                                        and right_label.swap_upper_n == 0):
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
                        if (isinstance(label, ODE_DAG_Repository.Swap) and label.swap_upper_n == 0):
                            n = label.with_lower_m
                            if len(r) == 3 and r[0] is not None:
                                right_label, right_i, right_o = r
                                if (isinstance(right_label, ODE_DAG_Repository.Swap)
                                        and right_label.swap_upper_n == 0):
                                    m = right_label.with_lower_m
                                    """
                                    beside(swap(n, 0, n), swap(m, 0, m))
                                    ->
                                    swap(n+m, 0, n+m)
                                    """
                                    # i < len(new_value) is an invariant, because len(zip(value[:-1], value[1:])) == len(value) - 1
                                    before_i = new_value[:index]
                                    after_i = new_value[index+2:]
                                    new_value = before_i + ((ODE_DAG_Repository.Swap(0, n + m), n + m, n + m),) + after_i
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
            # TODO: DAG Criteria: At most one edge between two nodes?
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
                        if isinstance(label, ODE_DAG_Repository.Swap):
                            m = label.swap_upper_n
                            n = label.with_lower_m
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
                                if isinstance(right_label, ODE_DAG_Repository.Swap):
                                    right_m = right_label.swap_upper_n
                                    right_n = right_label.with_lower_m
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
                        if isinstance(label, ODE_DAG_Repository.Swap):
                            if label.swap_upper_n == 0:
                                return False
                    if len(l) == 2 and len(r) == 2:
                        left_first, left_second = l
                        right_first, right_second = r
                        if left_first is not None and left_second is not None and right_first is not None and right_second is not None:
                            label_l_1, i_l_1, o_l_1 = left_first
                            label_l_2, i_l_2, o_l_2 = left_second
                            label_r_1, i_r_1, o_r_1 = right_first
                            label_r_2, i_r_2, o_r_2 = right_second
                            if isinstance(label_l_1, ODE_DAG_Repository.Swap):
                                m = label_l_1.swap_upper_n
                                n = label_l_1.with_lower_m
                                if isinstance(label_l_2, ODE_DAG_Repository.Swap) and label_l_2.swap_upper_n == 0:
                                    p = label_l_2.with_lower_m
                                    if isinstance(label_r_1, ODE_DAG_Repository.Swap) and label_r_1.swap_upper_n == 0:
                                        right_n = label_r_1.with_lower_m
                                        if isinstance(label_r_2, ODE_DAG_Repository.Swap):
                                            right_m = label_r_2.swap_upper_n
                                            right_p = label_r_2.with_lower_m
                                            if m is not None and n is not None and p is not None and right_m is not None and right_n is not None and right_p is not None:
                                                if m == right_m and n == right_n and p == right_p:
                                                    """
                                                    before(besides(swap(m+n, m, n), copy(p,edge())), besides(copy(n, edge()), swap(m+p, m, p)))
                                                    ->
                                                    swap(m + n + p, m, n+p)
                                                    """
                                                    return False
                            if isinstance(label_l_1, ODE_DAG_Repository.Swap) and label_l_1.swap_upper_n == 0:
                                m = label_l_1.with_lower_m
                                if isinstance(label_l_2, ODE_DAG_Repository.Swap):
                                    n = label_l_2.swap_upper_n
                                    p = label_l_2.with_lower_m
                                    if isinstance(label_r_1, ODE_DAG_Repository.Swap):
                                        right_m = label_r_1.swap_upper_n
                                        right_p = label_r_1.with_lower_m
                                        if isinstance(label_r_2, ODE_DAG_Repository.Swap) and label_r_2.swap_upper_n == 0:
                                            right_n = label_r_2.with_lower_m
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
                        if isinstance(left_label, ODE_DAG_Repository.Swap):
                            left_m = left_label.swap_upper_n
                            left_n = left_label.with_lower_m
                            if len(m) == 2 and len(r) == 1 and r[0] is not None:
                                mid_first, mid_second = m
                                right_label, right_i, right_o = r[0]
                                if mid_first is not None and mid_second is not None:
                                    mid_first_label, mid_n, mid_p = mid_first
                                    mid_second_label, mid_m, mid_q = mid_second
                                    if isinstance(right_label, ODE_DAG_Repository.Swap):
                                        right_p = right_label.swap_upper_n
                                        right_q = right_label.with_lower_m
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
                        if isinstance(label, ODE_DAG_Repository.Swap):
                            m = label.swap_upper_n
                            n = label.with_lower_m
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
                                if isinstance(right_label, ODE_DAG_Repository.Swap):
                                    right_m = right_label.swap_upper_n
                                    right_n = right_label.with_lower_m
                                    if m == right_n and n == right_m:
                                        before_i = new_value[:index]
                                        after_i = new_value[index + 2:]
                                        new_value = before_i + (((ODE_DAG_Repository.Swap(0, n + m), n + m, n + m),),) + after_i
                                        break
                    if len(r) == 1 and r[0] is not None:
                        """
                        before(x, edge())
                        ->
                        x
                        """
                        label, i, o = r[0]
                        if isinstance(label, ODE_DAG_Repository.Swap):
                            if label.swap_upper_n == 0:
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
                            if isinstance(label_l_1, ODE_DAG_Repository.Swap):
                                m = label_l_1.swap_upper_n
                                n = label_l_1.with_lower_m
                                if (isinstance(label_l_2, ODE_DAG_Repository.Swap) and label_l_2.swap_upper_n == 0):
                                    p = label_l_2.with_lower_m
                                    if (isinstance(label_r_1, ODE_DAG_Repository.Swap) and label_r_1.swap_upper_n == 0):
                                        right_n = label_r_1.with_lower_m
                                        if isinstance(label_r_2, ODE_DAG_Repository.Swap):
                                            right_m = label_r_2.swap_upper_n
                                            right_p = label_r_2.with_lower_m
                                            if m == right_m and n == right_n and p == right_p:
                                                """
                                                before(besides(swap(m+n, m, n), copy(p,edge())), besides(copy(n, edge()), swap(m+p, m, p)))
                                                ->
                                                swap(m + n + p, m, n+p)
                                                """
                                                before_i = new_value[:index]
                                                after_i = new_value[index + 2:]
                                                new_value = before_i + (((ODE_DAG_Repository.Swap(m, n + p), m + n + p, m + n + p),),) + after_i
                                                break
                            if (isinstance(label_l_1, ODE_DAG_Repository.Swap) and label_l_1.swap_upper_n == 0):
                                m = label_l_1.with_lower_m
                                if isinstance(label_l_2, ODE_DAG_Repository.Swap):
                                    n = label_l_2.swap_upper_n
                                    p = label_l_2.with_lower_m
                                    if isinstance(label_r_1, ODE_DAG_Repository.Swap):
                                        right_m = label_r_1.swap_upper_n
                                        right_p = label_r_1.with_lower_m
                                        if (isinstance(label_r_2, ODE_DAG_Repository.Swap) and label_r_2.swap_upper_n == 0):
                                            right_n = label_r_2.with_lower_m
                                            if m == right_m and n == right_n and p == right_p:
                                                """
                                                before(besides(copy(m, edge()), swap(n+p, n, p)), besides(swap(m+p, m, p), copy(n,edge())))
                                                ->
                                                swap(m + n + p, m+n, p)
                                                """
                                                before_i = new_value[:index]
                                                after_i = new_value[index + 2:]
                                                new_value = before_i + (((ODE_DAG_Repository.Swap(m + n, p), m + n + p, m + n + p),),) + after_i
                                                break
                for l, m, r in zip(value[:-2], value[1:-1], value[2:]):
                    if len(l) == 1:
                        left_label, left_i, left_o = l[0]
                        if isinstance(left_label, ODE_DAG_Repository.Swap):
                            left_m = left_label.swap_upper_n
                            left_n = left_label.with_lower_m
                            if len(m) == 2 and len(r) == 1:
                                mid_first, mid_second = m
                                right_label, right_i, right_o = r[0]
                                mid_first_label, mid_n, mid_p = mid_first
                                mid_second_label, mid_m, mid_q = mid_second
                                if isinstance(right_label, ODE_DAG_Repository.Swap):
                                    right_p = right_label.swap_upper_n
                                    right_q = right_label.with_lower_m
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

    @dataclass(frozen=True)
    class MSEloss:
        reduction: str | None = "mean"

    class Loss_Function(Group):
        name = "Loss_Function"

        def __init__(self):
            pass

        def __iter__(self):
            yield ODE_DAG_Repository.MSEloss(reduction="mean")
            yield ODE_DAG_Repository.MSEloss(reduction="sum")
            yield ODE_DAG_Repository.MSEloss(reduction="none")

        def __contains__(self, value):
            return value is None or (isinstance(value, ODE_DAG_Repository.MSEloss) and (value.reduction is None or value.reduction in ["mean", "sum", "none"]))

    @dataclass(frozen=True)
    class Adam:
        learning_rate: float | None = 0.001
        betas: tuple[float, float] | None = (0.9, 0.999)
        eps: float | None = 1e-08
        weight_decay: float | None = 0.0
        amsgrad: bool | None = False
        maximize: bool | None = False
        capturable: bool | None = False
        differentiable: bool | None = False
        decoupled_weight_decay: bool | None = False

    class Optimizer(Group):
        name = "Optimizer"

        def __init__(self, learning_rate_values):
            self.learning_rate_values = learning_rate_values

        def __iter__(self):
            for lr in self.learning_rate_values:
                yield ODE_DAG_Repository.Adam(learning_rate=lr)

        def __contains__(self, value):
            return value is None or (isinstance(value, ODE_DAG_Repository.Adam) and (value.learning_rate is None or value.learning_rate in self.learning_rate_values))




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
            if (left_term_root == "swap") and "beside_cons" in right_head_root and "before_singleton" in right_tail_root:
                if len(left_term.children) != 4 or len(right_head.children) != 11 or len(right_tail.children) != 6:
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
                        if len(right_swap.children) != 4:
                            raise ValueError("Derivation trees have not the expected shape.")
                        p = right_swap.children[1]
                        q = right_swap.children[2]
                        if m == y_m and n == x_n and p == x_p and q == y_q:
                            return False
            elif (left == "swap") and "beside_cons" in right_head_root and "before_cons" in right_tail_root:
                if len(left_term.children) != 4 or len(right_head.children) != 11 or len(right_tail.children) != 9:
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
                        if len(right_swap.children) != 4:
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
                if len(left_head.children) != 4 or len(left_tail.children) != 5 or len(right_term.children) != 11:
                    raise ValueError("Derivation trees have not the expected shape.")
                m = left_head.children[1]
                n = left_head.children[2]
                left_tail_term = left_tail.children[4] # swap(p, 0, p)
                right_head = right_term.children[9] # swap(n, 0, n)
                right_tail = right_term.children[10]

                left_tail_swap = left_tail_term.root
                right_head_root = right_head.root
                right_tail_root = right_tail.root
                if left_tail_swap == "edges" and right_head_root == "edges" and "beside_singleton" in right_tail_root:
                    if len(left_tail_term.children) != 2 or len(right_head.children) != 2 or len(right_tail.children) != 5:
                        raise ValueError("Derivation trees have not the expected shape.")
                    p = left_tail_term.children[0]
                    right_n = right_head.children[0]

                    right_tail_term = right_tail.children[4] # swap(m+p, m, p)
                    right_tail_term_root = right_tail_term.root
                    if (right_tail_term_root == "swap") and n == right_n:
                        if len(right_tail_term.children) != 4:
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
                if len(left_head.children) != 4 or len(left_tail.children) != 5 or len(right_term.children) != 11:
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
                    if len(left_tail_term.children) != 2 or len(right_head.children) != 2 or len(
                            right_tail.children) != 5:
                        raise ValueError("Derivation trees have not the expected shape.")
                    p = left_tail_term.children[0]
                    right_n = right_head.children[0]

                    right_tail_term = right_tail.children[4]  # swap(m+p, m, p)
                    right_tail_term_root = right_tail_term.root
                    if right_tail_term_root == "swap" and n == right_n:
                        if len(right_tail_term.children) != 4:
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
                if len(left_term.children) != 4 or len(right_term.children) != 5:
                    raise ValueError("Derivation trees have not the expected shape.")
                m = left_term.children[1]
                n = left_term.children[2]

                right_beside = right_term.children[4]
                right_beside_root = right_beside.root
                if right_beside_root == "swap" or right_beside_root == "swap":
                    if len(right_beside.children) != 4:
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
                if len(left_term.children) != 4 or len(right_term.children) != 5:
                    raise ValueError("Derivation trees have not the expected shape.")
                m = left_term.children[1]
                n = left_term.children[2]

                right_beside = right_term.children[4]
                right_beside_root = right_beside.root
                if right_beside_root == "swap":
                    if len(right_beside.children) != 4:
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
                if len(left_head.children) != 2 or len(left_tail.children) != 5 or len(right_term.children) != 11:
                    raise ValueError("Derivation trees have not the expected shape.")
                m = left_head.children[0]

                left_tail_term = left_tail.children[4]
                right_head = right_term.children[9]
                right_tail = right_term.children[10]

                left_tail_term_root = left_tail_term.root
                right_head_root = right_head.root
                right_tail_root = right_tail.root
                if left_tail_term_root == "swap" and right_head_root == "swap" and "beside_singleton" in right_tail_root:
                    if len(left_tail_term.children) != 4 or len(right_head.children) != 4 or len(right_tail.children) != 5:
                        raise ValueError("Derivation trees have not the expected shape.")
                    n = left_tail_term.children[1]
                    p = left_tail_term.children[2]

                    right_m = right_head.children[1]
                    right_p = right_head.children[2]

                    right_tail_term = right_tail.children[4]
                    right_tail_term_root = right_tail_term.root

                    if right_tail_term_root == "edges" and m == right_m and p == right_p:
                        if len(right_tail_term.children) != 2:
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
                if len(left_head.children) != 2 or len(left_tail.children) != 5 or len(right_term.children) != 11:
                    raise ValueError("Derivation trees have not the expected shape.")
                m = left_head.children[0]

                left_tail_term = left_tail.children[4]
                right_head = right_term.children[9]
                right_tail = right_term.children[10]

                left_tail_term_root = left_tail_term.root
                right_head_root = right_head.root
                right_tail_root = right_tail.root
                if left_tail_term_root == "swap" and right_head_root == "swap" and "beside_singleton" in right_tail_root:
                    if len(left_tail_term.children) != 4 or len(right_head.children) != 4 or len(
                            right_tail.children) != 5:
                        raise ValueError("Derivation trees have not the expected shape.")
                    n = left_tail_term.children[1]
                    p = left_tail_term.children[2]

                    right_m = right_head.children[1]
                    right_p = right_head.children[2]

                    right_tail_term = right_tail.children[4]
                    right_tail_term_root = right_tail_term.root

                    if right_tail_term_root == "edges" and m == right_m and p == right_p:
                        if len(right_tail_term.children) != 2:
                            raise ValueError("Derivation trees have not the expected shape.")
                        right_n = right_tail_term.children[0]
                        if n == right_n:
                            return False
        return True

    def dag_constraint(self, head: DerivationTree[Any, str, Any], tail: DerivationTree[Any, str, Any], i : int) -> bool:
        x = head.interpret(self.edgelist_algebra())
        y = tail.interpret(self.edgelist_algebra())
        id = (0,0)
        inputs = ["input" for _ in range(0,i)]
        edgelist = y[0]((id[0] + 2.5, id[1]), x(id, inputs)[1])[0]
        edgeset = set(edgelist)
        return len(edgelist) == len(edgeset)


    def specification(self):
        labels = self.Label(self.dimensions, self.linear_feature_dimensions, self.sharpness_values, self.threshold_values)
        para_labels = self.Para(labels, self.dimensions)
        paratuples = self.ParaTuples(para_labels, max_length=max(self.dimensions))
        paratupletuples = self.ParaTupleTuples(paratuples)
        dimension = DataGroup("dimension", self.dimensions)
        dimension_with_None = DataGroup("dimension_with_None", list(self.dimensions) + [None])
        linear_feature_dimension = DataGroup("linear_feature_dimension", self.linear_feature_dimensions)
        sharpness_values = DataGroup("sharpness_values", self.sharpness_values)
        threshold_values = DataGroup("threshold_values", self.threshold_values)
        bool = DataGroup("bool", [True, False])
        learning_rate = DataGroup("learning_rate", self.learning_rate_values)
        loss_function = ODE_DAG_Repository.Loss_Function()
        adam_learning_rate = DataGroup("adam_learning_rate", self.adam_learning_rate_values)
        optimizer = ODE_DAG_Repository.Optimizer(self.adam_learning_rate_values)
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
            .parameter("para", para_labels, lambda v: [(self.Swap(0, v["io"]), v["io"], v["io"]),
                                                       (self.Swap(0, None), v["io"], v["io"]),
                                                       (self.Swap(None, v["io"]), v["io"], v["io"]),
                                                       (self.Swap(None, None), v["io"], v["io"]),
                                                       (self.Swap(0, v["io"]), v["io"], None),
                                                       (self.Swap(0, None), v["io"], None),
                                                       (self.Swap(None, v["io"]), v["io"], None),
                                                       (self.Swap(None, None), v["io"], None),
                                                       (self.Swap(0, v["io"]), None, v["io"]),
                                                       (self.Swap(0, None), None, v["io"]),
                                                       (self.Swap(None, v["io"]), None, v["io"]),
                                                       (self.Swap(None, None), None, v["io"]),
                                                       (self.Swap(0, v["io"]), None, None),
                                                       (self.Swap(0, None), None, None),
                                                       (self.Swap(None, v["io"]), None, None),
                                                       (self.Swap(None, None), None, None), None])
                                                        # (None, None, None)])
            .suffix(Constructor("Model_component", Constructor("input", Var("io"))
                                & Constructor("input", Literal(None))
                                & Constructor("output", Var("io"))
                                & Constructor("output", Literal(None))
                                & Constructor("structure", Var("para"))
                                & Constructor("structure", Literal(None))
                                ) & Constructor("ID")
                    ),

            "swap": SpecificationBuilder()
            .parameter("io", dimension)
            .parameter("n", dimension, lambda v: range(1, v["io"]))
            .parameter("m", dimension, lambda v: [v["io"] - v["n"]]) # m > 0
            .parameter("para", para_labels, lambda v: [(self.Swap(v["n"], v["m"]), v["io"], v["io"]),
                                                       (self.Swap(v["n"], None), v["io"], v["io"]),
                                                       (self.Swap(None, v["m"]), v["io"], v["io"]),
                                                       (self.Swap(None, None), v["io"], v["io"]),
                                                       (self.Swap(v["n"], v["m"]), v["io"], None),
                                                       (self.Swap(v["n"], None), v["io"], None),
                                                       (self.Swap(None, v["m"]), v["io"], None),
                                                       (self.Swap(None, None), v["io"], None),
                                                       (self.Swap(v["n"], v["m"]), None, v["io"]),
                                                       (self.Swap(v["n"], None), None, v["io"]),
                                                       (self.Swap(None, v["m"]), None, v["io"]),
                                                       (self.Swap(None, None), None, v["io"]),
                                                       (self.Swap(v["n"], v["m"]), None, None),
                                                       (self.Swap(v["n"], None), None, None),
                                                       (self.Swap(None, v["m"]), None, None),
                                                       (self.Swap(None, None), None, None), None])
                                                       #(None, None, None)])
            .suffix(Constructor("Model_component", Constructor("input", Var("io"))
                                & Constructor("input", Literal(None))
                                & Constructor("output", Var("io"))
                                & Constructor("output", Literal(None))
                                & Constructor("structure", Var("para"))
                                & Constructor("structure", Literal(None))
                                ) & Constructor("non_ID")
                    ),

            "linear": SpecificationBuilder()
            .parameter("in_f", linear_feature_dimension)
            .parameter("out_f", linear_feature_dimension)
            .parameter("bias", bool)
            .parameter("i", dimension)
            .parameter("o", dimension)
            .parameter("para", para_labels, lambda v: [(self.Linear(in_f, out_f, bias), i, o)
                                                       for in_f in [v["in_f"], None]
                                                       for out_f in [v["out_f"], None]
                                                       for bias in [v["bias"], None]
                                                       for i in [v["i"], None]
                                                       for o in [v["o"], None]] + [None])
            .suffix(Constructor("Model_component",
                                Constructor("input", Var("i"))
                                & Constructor("input", Literal(None))
                                & Constructor("output", Var("o"))
                                & Constructor("output", Literal(None))
                                & Constructor("structure", Var("para"))
                                & Constructor("structure", Literal(None))
                                ) & Constructor("non_ID")
                    ),

            "sigmoid": SpecificationBuilder()
            .parameter("i", dimension)
            .parameter("o", dimension)
            .parameter("para", para_labels, lambda v: [(self.Sigmoid(), v["i"], v["o"]), None])
            .suffix(Constructor("Model_component",
                                Constructor("input", Var("i"))
                                & Constructor("input", Literal(None))
                                & Constructor("output", Var("o"))
                                & Constructor("output", Literal(None))
                                & Constructor("structure", Var("para"))
                                & Constructor("structure", Literal(None))
                                ) & Constructor("non_ID")
                    ),

            "sharpness_sigmoid": SpecificationBuilder()
            .parameter("sharpness", sharpness_values)
            .parameter("i", dimension)
            .parameter("o", dimension)
            .parameter("para", para_labels, lambda v: [(self.Sharpness_Sigmoid(sharpness), i, o)
                                                       for sharpness in [v["sharpness"], None]
                                                       for i in [v["i"], None]
                                                       for o in [v["o"], None]] + [None])
            .suffix(Constructor("Model_component",
                                Constructor("input", Var("i"))
                                & Constructor("input", Literal(None))
                                & Constructor("output", Var("o"))
                                & Constructor("output", Literal(None))
                                & Constructor("structure", Var("para"))
                                & Constructor("structure", Literal(None))
                                ) & Constructor("non_ID")
                    ),

            "relu": SpecificationBuilder()
            .parameter("inplace", bool)
            .parameter("i", dimension)
            .parameter("o", dimension)
            .parameter("para", para_labels, lambda v: [(self.ReLu(b), i, o)
                                                       for b in [v["inplace"], None]
                                                       for i in [v["i"], None]
                                                       for o in [v["o"], None]] + [None])
            .suffix(Constructor("Model_component",
                                Constructor("input", Var("i"))
                                & Constructor("input", Literal(None))
                                & Constructor("output", Var("o"))
                                & Constructor("output", Literal(None))
                                & Constructor("structure", Var("para"))
                                & Constructor("structure", Literal(None))
                                ) & Constructor("non_ID")
                    ),

            "lte": SpecificationBuilder()
            .parameter("threshold", threshold_values)
            .parameter("i", dimension)
            .parameter("o", dimension)
            .parameter("para", para_labels, lambda v: [(self.lte(t), i, o)
                                                       for t in [v["threshold"], None]
                                                       for i in [v["i"], None]
                                                       for o in [v["o"], None]] + [None])
            .suffix(Constructor("Model_component",
                                Constructor("input", Var("i"))
                                & Constructor("input", Literal(None))
                                & Constructor("output", Var("o"))
                                & Constructor("output", Literal(None))
                                & Constructor("structure", Var("para"))
                                & Constructor("structure", Literal(None))
                                ) & Constructor("non_ID")
                    ),

            "join": SpecificationBuilder()
            .parameter("i", dimension)
            .parameter_constraint(lambda v: v["i"] > 1)
            .parameter("o", dimension)
            .parameter("para", para_labels, lambda v: [(self.Join(), v["i"], v["o"]), None])
            .suffix(Constructor("Model_component",
                                Constructor("input", Var("i"))
                                & Constructor("input", Literal(None))
                                & Constructor("output", Var("o"))
                                & Constructor("output", Literal(None))
                                & Constructor("structure", Var("para"))
                                & Constructor("structure", Literal(None))
                                ) & Constructor("non_ID")
                    ),

            #'''
            "beside_singleton": SpecificationBuilder()
            .parameter("i", dimension)
            .parameter("o", dimension)
            .parameter("ls", paratuples)
            .parameter_constraint(lambda v: v["ls"] is not None and len(v["ls"]) == 1)
            .parameter("para", para_labels, lambda v: [v["ls"][0]])
            .parameter_constraint(lambda v: (len(v["para"]) == 3 and
                                             (v["para"][1] == v["i"] if v["para"][1] is not None else True) and
                                             (v["para"][2] == v["o"] if v["para"][2] is not None else True)
                                             ) if v["para"] is not None else True)
            .suffix(
                ((Constructor("Model_component",
                                       Constructor("input", Var("i"))
                                       & Constructor("output", Var("o"))
                                       & Constructor("structure", Var("para")))
                     & Constructor("non_ID")
                  )
                 **
                 (Constructor("Model_parallel",
                                Constructor("input", Var("i"))
                                & Constructor("input", Literal(None))
                                & Constructor("output", Var("o"))
                                & Constructor("output", Literal(None))
                                & Constructor("structure", Var("ls"))
                                & Constructor("structure", Literal(None))
                                )
                  & Constructor("non_ID") & Constructor("last", Constructor("non_ID"))
                  )
                 )
                &
                ((Constructor("Model_component",
                                       Constructor("input", Var("i"))
                                       & Constructor("output", Var("o"))
                                       & Constructor("structure", Var("para")))
                     & Constructor("ID")
                     )
                 **
                 (Constructor("Model_parallel",
                                Constructor("input", Var("i"))
                                & Constructor("input", Literal(None))
                                & Constructor("output", Var("o"))
                                & Constructor("output", Literal(None))
                                & Constructor("structure", Var("ls"))
                                & Constructor("structure", Literal(None))
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
            .parameter_constraint(lambda v: v["ls"] is not None and len(v["ls"]) > 1)
            .parameter("head", para_labels, lambda v: [v["ls"][0]])
            .parameter_constraint(lambda v: v["head"] is None or (len(v["head"]) == 3 and
                                                                  (v["head"][1] == v["i1"] or v["head"][1] is None) and
                                                                  (v["head"][2] == v["o1"] or v["head"][2] is None)))
            .parameter("tail", paratuples, lambda v: [v["ls"][1:]])
            .suffix(
                    ((Constructor("Model_component",
                                       Constructor("input", Var("i1"))
                                       & Constructor("output", Var("o1"))
                                       & Constructor("structure", Var("head")))
                      & Constructor("ID"))
                    **
                    (Constructor("Model_parallel",
                                       Constructor("input", Var("i2"))
                                       & Constructor("output", Var("o2"))
                                       & Constructor("structure", Var("tail")))
                     & Constructor("non_ID") & Constructor("last", Constructor("non_ID")))
                     **
                     (Constructor("Model_parallel",
                                Constructor("input", Var("i"))
                                & Constructor("input", Literal(None))
                                & Constructor("output", Var("o"))
                                & Constructor("output", Literal(None))
                                & Constructor("structure", Var("ls"))
                                & Constructor("structure", Literal(None))
                                )
                      & Constructor("non_ID") & Constructor("last", Constructor("ID")))
                     )
                    &
                    ((Constructor("Model_component",
                                  Constructor("input", Var("i1"))
                                  & Constructor("output", Var("o1"))
                                  & Constructor("structure", Var("head")))
                      & Constructor("non_ID"))
                     **
                     (Constructor("Model_parallel",
                                  Constructor("input", Var("i2"))
                                  & Constructor("output", Var("o2"))
                                  & Constructor("structure", Var("tail")))
                      & Constructor("ID"))
                     **
                     (Constructor("Model_parallel",
                                  Constructor("input", Var("i"))
                                  & Constructor("input", Literal(None))
                                  & Constructor("output", Var("o"))
                                  & Constructor("output", Literal(None))
                                  & Constructor("structure", Var("ls"))
                                  & Constructor("structure", Literal(None))
                                  )
                      & Constructor("non_ID") & Constructor("last", Constructor("non_ID")))
                     )
                    &
                    ((Constructor("Model_component",
                                  Constructor("input", Var("i1"))
                                  & Constructor("output", Var("o1"))
                                  & Constructor("structure", Var("head")))
                      & Constructor("non_ID"))
                     **
                     (Constructor("Model_parallel",
                                  Constructor("input", Var("i2"))
                                  & Constructor("output", Var("o2"))
                                  & Constructor("structure", Var("tail")))
                      & Constructor("non_ID"))
                     **
                     (Constructor("Model_parallel",
                                  Constructor("input", Var("i"))
                                  & Constructor("input", Literal(None))
                                  & Constructor("output", Var("o"))
                                  & Constructor("output", Literal(None))
                                  & Constructor("structure", Var("ls"))
                                  & Constructor("structure", Literal(None))
                                  )
                      & Constructor("non_ID") & Constructor("last", Constructor("non_ID")))
                     )
                    ),

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
            .argument("x", Constructor("Model_parallel",
                                       Constructor("input", Var("i"))
                                       & Constructor("output", Var("o"))
                                       & Constructor("structure", Var("ls1"))) & Constructor("non_ID"))
            .suffix(Constructor("Model",
                                Constructor("input", Var("i"))
                                & Constructor("input", Literal(None))
                                & Constructor("output", Var("o"))
                                & Constructor("output", Literal(None))
                                & Constructor("structure", Var("request"))
                                )),
            # TODO: add constraint for sequential composition of linear layers
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
            .argument("x", Constructor("Model_parallel",
                                       Constructor("input", Var("i"))
                                       & Constructor("output", Var("j"))
                                       & Constructor("structure", Var("head"))) & Constructor("non_ID"))
            .argument("y", Constructor("Model",
                                       Constructor("input", Var("j"))
                                       & Constructor("output", Var("o"))
                                       & Constructor("structure", Var("tail"))))
            .constraint(lambda v: self.swaplaw1(v["x"], v["y"]))
            .constraint(lambda v: self.swaplaw2(v["x"], v["y"]))
            .constraint(lambda v: self.swaplaw3(v["x"], v["y"]))
            .constraint(lambda v: self.swaplaw4(v["x"], v["y"]))
            .constraint(lambda v: self.dag_constraint(v["x"], v["y"], v["i"])) # DAG: At most one edge between any two nodes
            .suffix(Constructor("Model",
                                Constructor("input", Var("i"))
                                & Constructor("input", Literal(None))
                                & Constructor("output", Var("o"))
                                & Constructor("output", Literal(None))
                                & Constructor("structure", Var("request")))),

            "mse_loss": SpecificationBuilder()
            .parameter("loss", loss_function, lambda v: [self.MSEloss()])
            .suffix(Constructor("Loss", Constructor("type", Var("loss")))),

            "adam_optimizer": SpecificationBuilder()
            .parameter("learning_rate", adam_learning_rate)
            .parameter("optimizer", optimizer, lambda v: [self.Adam(v["learning_rate"])])
            .suffix(Constructor("Optimizer", Constructor("type", Var("optimizer")))),

            "learner": SpecificationBuilder()
            .parameter("i", dimension_with_None)
            .parameter("o", dimension_with_None)
            .parameter("request", paratupletuples)
            .parameter("epochs", epochs)
            .parameter("learning_rate", learning_rate)
            .parameter("loss", loss_function, lambda v: [self.MSEloss()])
            .parameter("opti", optimizer, lambda v: [self.Adam(v["learning_rate"])])
            .argument("loss_f", Constructor("Loss", Constructor("type", Var("loss"))))
            .argument("optimizer", Constructor("Optimizer", Constructor("type", Var("opti"))))
            .argument("model", Constructor("Model",
                                           Constructor("input", Var("i"))
                                           & Constructor("output", Var("o"))
                                           & Constructor("structure", Var("request"))))
            .suffix(Constructor("Learner", Constructor("Model",
                                                       Constructor("input", Var("i"))
                                                       & Constructor("output", Var("o"))
                                                       & Constructor("structure", Var("request"))
                                                       )
                                & Constructor("Loss", Constructor("type", Var("loss")))
                                & Constructor("Optimizer", Constructor("type", Var("opti")))
                                & Constructor("learning_rate", Var("learning_rate"))
                                & Constructor("epochs", Var("epochs"))
                                )
                    )
        }

    # Interpretations of terms are algebras in my language

    def pretty_term_algebra(self):
        return {
            "edges": (lambda io, para: f"edges({io})"),

            "swap": (lambda io, n, m, para: f"swap({io}, {n}, {m})"),

            "linear": (lambda in_f, out_f, bias, i, o, para: str(para)),

            "sigmoid": (lambda i, o, para: str(para)),

            "sharpness_sigmoid": (lambda s, i, o, para: str(para)),

            "relu": (lambda inplace, i, o, para: str(para)),

            "lte": (lambda t, i, o, para: str(para)),

            "join": (lambda i, o, para: str(para)),

            "beside_singleton": (lambda i, o, ls, para, x: f"{x})"),

            "beside_cons": (lambda i, i1, i2, o, o1, o2, ls, head, tail, x, y: f"{x} || {y}"),

            "before_singleton": (lambda i, o, r, ls, ls1, x: f"({x}"),

            "before_cons": (lambda i, j, o, r, ls, head, tail, x, y: f"({x} ; {y}"),

            "mse_loss": (lambda l: str(l)),

            "adam_optimizer": (lambda lr, o: str(o)),

            "learner": (lambda i, o, r, e, lr, l, opt, loss, optimizer, model: f"""
Learner(
    model= (
        {model}
        ), 
    loss= {loss}, 
    optimizer= {optimizer}, 
    learning_rate= {lr}, 
    epochs= {e}
    )
"""),
        }

    # algebra for DAG-criteria and plotting
    # TODO: update to relabeling nodes with their combinator names
    def edgelist_algebra(self):
        return {
            "edges": (lambda io, para: lambda id, inputs: ([], inputs, {})),

            "swap": (lambda io, n, m, para: lambda id, inputs: ([], inputs[n:] + inputs[:n], {})),

            "node": (lambda l, i, o, para: lambda id, inputs: ([(x,l + str(id)) for x in inputs],  [l + str(id) for _ in range(0,o)], {l + str(id) : id})),

            "linear": (lambda in_f, out_f, bias, i, o, para: lambda id, inputs: ([(x, str(para) + str(id)) for x in inputs],  [str(para) + str(id) for _ in range(0,o)], {str(para) + str(id): id})),

            "sigmoid": (lambda i, o, para: lambda id, inputs: ([(x, str(para) + str(id)) for x in inputs],  [str(para) + str(id) for _ in range(0,o)], {str(para) + str(id): id})),

            "sharpness_sigmoid": (lambda s, i, o, para: lambda id, inputs: ([(x, str(para) + str(id)) for x in inputs],  [str(para) + str(id) for _ in range(0,o)], {str(para) + str(id): id})),

            "relu": (lambda inplace, i, o, para: lambda id, inputs: ([(x, str(para) + str(id)) for x in inputs],  [str(para) + str(id) for _ in range(0,o)], {str(para) + str(id): id})),

            "lte": (lambda t, i, o, para: lambda id, inputs: ([(x, str(para) + str(id)) for x in inputs],  [str(para) + str(id) for _ in range(0,o)], {str(para) + str(id): id})),

            "join": (lambda i, o, para: lambda id, inputs: ([(x, str(para) + str(id)) for x in inputs],  [str(para) + str(id) for _ in range(0,o)], {str(para) + str(id): id})),

            "beside_singleton": (lambda i, o, ls, para, x: x),

            "beside_cons": (lambda i, i1, i2, o, o1, o2, ls, head, tail, x, y: lambda id, inputs:
                (x(id, inputs[:i1])[0] + y((id[0], id[1] + 0.2), inputs[i1:])[0],
                 x(id, inputs[:i1])[1] + y((id[0], id[1] + 0.2), inputs[i1:])[1],
                 x(id, inputs[:i1])[2] | y((id[0], id[1] + 0.2), inputs[i1:])[2])),

            "before_singleton": (lambda i, o, r, ls, ls1, x: (x, i)),

            "before_cons": (lambda i, j, o, r, ls, head, tail, x, y: (lambda id, inputs:
                                                                      (y[0]((id[0] + 2.5, id[1]), x(id, inputs)[1])[0] + x(id, inputs)[0],
                                                                       y[0]((id[0] + 2.5, id[1]), x(id, inputs)[1])[1],
                                                                       y[0]((id[0] + 2.5, id[1]), x(id, inputs)[1])[2] | x(id, inputs)[2]),
                                                                      i)),
            "mse_loss": (lambda l: ()),

            "adam_optimizer": (lambda lr, o: ()),

            "learner": (lambda i, o, r, e, lr, l, opt, loss, optimizer, model: model)
        }


    def pytorch_code_algebra(self):
        return {
            "edges": (lambda io, para: lambda id: (f"""""", f"""""")),

            "swap": (lambda io, n, m, para: lambda id: (f"""""", f"""""")),

            "linear": (lambda in_f, out_f, bias, i, o, para: lambda id: (f"""
            self.linear_{id} = nn.Linear({in_f}, {out_f}, bias={bias})
""", lambda x: f"self.linear_{id}({x})")),

            "sigmoid": (lambda i, o, para: lambda id: (f"", lambda x: f"torch.sigmoid({x})")),

            "sharpness_sigmoid": (lambda s, i, o, para: lambda id: (f"""
            self.sharpness_{id} = {s}
""", lambda x: f"torch.sigmoid(-self.sharpness_{id} * {x})")),

            "relu": (lambda inplace, i, o, para: lambda id: (f"""
            self.relu_{id} = nn.ReLU(inplace={inplace})
""", lambda x: f"self.relu_{id}({x})")),

            "lte": (lambda t, i, o, para: lambda id: (f"", lambda x: f"({x} <= 0).float()")),

            "join": (lambda i, o, para: lambda id: (f"", f"""""")),

            "beside_singleton": (lambda i, o, ls, para, x: lambda id: (f"""""", f"""""")),

            "beside_cons": (lambda i, i1, i2, o, o1, o2, ls, head, tail, x, y: lambda id: (f"""""", f"""""")),

            "before_singleton": (lambda i, o, r, ls, ls1, x: lambda id: (f"""""", f"""""")),

            "before_cons": (lambda i, j, o, r, ls, head, tail, x, y: lambda id: (f"""""", f"""""")),

            "mse_loss": (lambda l: lambda id: (f"""""", f"""""")),

            "adam_optimizer": (lambda lr, o: lambda id: (f"""""", f"""""")),

            "learner": (lambda i, o, r, e, lr, l, opt, loss, optimizer, model: lambda id: (f"""""", f"""""")),
            }




if __name__ == "__main__":
    repo = ODE_DAG_Repository(dimensions=range(1, 4), linear_feature_dimensions=range(1, 2), sharpness_values=[2],
                              threshold_values=[0], learning_rate_values=[1e-2], adam_learning_rate_values=[1e-2],
                              n_epoch_values=[10000])

    target0 = Constructor("Model_component",
                          Constructor("input", Literal(1))
                          & Constructor("output", Literal(1))
                          & Constructor("structure", Literal((repo.Linear(1, 1, True), 1, 1)))
                          )

    target1 = Constructor("Model",
                          Constructor("input", Literal(1))
                          & Constructor("output", Literal(1))
                          & Constructor("structure", Literal(
                              (((repo.Linear(1, 1, True), 1, 1),),
                               ((repo.ReLu(), 1, 1),),
                               ((repo.Linear(1, 1, True), 1, 1),))
                          ))
                          )

    target2 = Constructor("Model",
                          Constructor("input", Literal(3))
                          & Constructor("output", Literal(1))
                          & Constructor("structure", Literal(
                              (None, None, None)
                          ))
                          )

    target3 = Constructor("Learner", target1
                          & Constructor("Loss", Constructor("type", Literal(repo.MSEloss())))
                          & Constructor("Optimizer", Constructor("type", Literal(repo.Adam(1e-2))))
                          & Constructor("learning_rate", Literal(1e-2))
                          & Constructor("epochs", Literal(10000))
                          )

    target = target3
    from cl3s import SearchSpaceSynthesizer
    synthesizer = SearchSpaceSynthesizer(repo.specification(), {})

    print(target)

    search_space = synthesizer.construct_search_space(target).prune()
    print("finish synthesis, start enumerate")
    terms = search_space.enumerate_trees(target, 10)

    for t in terms:
        #print(t)
        print(t.interpret(repo.pretty_term_algebra()))
