from cosy import SpecificationBuilder, Constructor, Literal, Var, Synthesizer
from cosy.tree import Tree
from cosy.types import DataGroup, Group

# TODO: write normalization function for structure literals for an increased usability of the request language

class Labeled_DAMG_Repository:
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


            (the three laws below may be handled with types, but only if the contains-bug is solved!)
            (one could introduce a "only edges" flag, like ID and non_ID, but that would also require a lot of thinking...)

            before(besides(swap(m+n, m, n), copy(p,edge())), besides(copy(n, edge()), swap(m+p, m, p)))
            ->
            swap(m + n + p, m, n+p)

            before(swap(m+n, m, n), swap(n+m, n, m))
            ->
            copy(m+n, edge())

            before(besides(copy(m, edge()), swap(n+p, n, p)), besides(swap(m+p, m, p), copy(n,edge())))
            ->
            swap(m + n + p, m+n, p)


            additionally, we should satisfy the following law:
            (this should be doable with types, too)

            beside(swap(n, 0, n), swap(m, 0, m))
            ->
            swap(n+m, 0, n+m)

            """

    def __init__(self, labels, dimensions):
        # additionally to labeled nodes, we have (unlabelled) edges, that needs to be handled additionally
        if "swap" in labels:
            raise ValueError("Label 'swap' is reserved and cannot be used as a node label.")
        self.labels = labels
        self.dimensions = dimensions
        self.id_seed = 0

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
                                    #if i is not None and o is not None and i == o:
                                        for n in range (0, i):
                                            m = i - n
                                            assert m > 0
                                            yield ("swap", n, m), i, o
            yield None


        def __contains__(self, value):
            return value is None or ((isinstance(value, tuple) and len(value) == 3 and (value[0] in self.labels or
                                                                      (value[0][0] == "swap" and
                                                                       (value[0][1] in self.dimensions or value[0][1] == 0) and
                                                                       value[0][2] in self.dimensions))
                    and value[1] in self.dimensions and value[2] in self.dimensions))

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

    class ParaTupleTuples(Group):
        name = "ParaTupleTuples"

        def __init__(self, para_tuples):
            self.para_tuples = para_tuples

        def __iter__(self):
            return super().__iter__()

        def __contains__(self, value):
            return value is None or (isinstance(value, tuple) and all(True if v is None else v in self.para_tuples for v in value))

    @staticmethod
    def swaplaw1(head: Tree[str], tail: Tree[str]) -> bool:
        """
        before(swap(m+n, m, n), before(beside(x(n,p), y(m,q)), swap(p+q, p, q)))
        ->
        beside(y(m,q),x(n,p))

        forbid the pattern on the left-hand side of the rewrite rule by returning False if it is matched
        """

        left = head.root
        right = tail.root

        if "beside_singleton" in left and "before_cons" in right:
            if len(head.children) != 5 or len(tail.children) != 8:
                raise ValueError("Derivation trees have not the expected shape.")
            left_term = head.children[4]
            right_head = tail.children[6]
            right_tail = tail.children[7]

            left_term_root = left_term.root
            right_head_root = right_head.root
            right_tail_root = right_tail.root
            if (left_term_root == "swap") and "beside_cons" in right_head_root and "before_singleton" in right_tail_root:
                if len(left_term.children) != 4 or len(right_head.children) != 11 or len(right_tail.children) != 5:
                    raise ValueError("Derivation trees have not the expected shape.")
                m = left_term.children[1]
                n = left_term.children[2]
                x_n = right_head.children[1]
                x_p = right_head.children[4]
                right_head_tail = right_head.children[10]
                right_tail_term = right_tail.children[4]
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
                if len(left_term.children) != 4 or len(right_head.children) != 11 or len(right_tail.children) != 8:
                    raise ValueError("Derivation trees have not the expected shape.")
                m = left_term.children[1]
                n = left_term.children[2]
                x_n = right_head.children[1]
                x_p = right_head.children[4]
                right_head_tail = right_head.children[10]
                right_tail_head = right_tail.children[6]
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
    def swaplaw2(head: Tree[str], tail: Tree[str]) -> bool:
        """
        before(besides(swap(m+n, m, n), copy(p,edge())), besides(copy(n, edge()), swap(m+p, m, p)))
        ->
        swap(m + n + p, m, n+p)

        forbid the pattern on the left-hand side of the rewrite rule by returning False if it is matched

        """

        left = head.root
        right = tail.root

        if "beside_cons" in left and "before_singleton" in right:
            if len(head.children) != 11 or len(tail.children) != 5:
                raise ValueError("Derivation trees have not the expected shape.")
            left_head = head.children[9]
            left_tail = head.children[10]
            right_term = tail.children[4]

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
            if len(head.children) != 11 or len(tail.children) != 8:
                raise ValueError("Derivation trees have not the expected shape.")
            left_head = head.children[9]
            left_tail = head.children[10]
            right_term = tail.children[6]

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
    def swaplaw3(head: Tree[str], tail: Tree[str]) -> bool:
        """
        before(swap(m+n, m, n), swap(n+m, n, m))
        ->
        copy(m+n, edge())

        forbid the pattern on the left-hand side of the rewrite rule by returning False if it is matched
        """
        left = head.root
        right = tail.root

        if "beside_singleton" in left and "before_singleton" in right:
            if len(head.children) != 5 or len(tail.children) != 5:
                raise ValueError("Derivation trees have not the expected shape.")
            left_term = head.children[4]
            right_term = tail.children[4]


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
            if len(head.children) != 5 or len(tail.children) != 8:
                raise ValueError("Derivation trees have not the expected shape.")
            left_term = head.children[4]
            right_term = tail.children[6]


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
    def swaplaw4(head: Tree[str], tail: Tree[str]) -> bool:
        """
        before(besides(copy(m, edge()), swap(n+p, n, p)), besides(swap(m+p, m, p), copy(n,edge())))
        ->
        swap(m + n + p, m+n, p)

        forbid the pattern on the left-hand side of the rewrite rule by returning False if it is matched
        """
        left = head.root
        right = tail.root

        if "beside_cons" in left and "before_singleton" in right:
            if len(head.children) != 11 or len(tail.children) != 5:
                raise ValueError("Derivation trees have not the expected shape.")
            left_head = head.children[9]
            left_tail = head.children[10]
            right_term = tail.children[4]

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
            if len(head.children) != 11 or len(tail.children) != 8:
                raise ValueError("Derivation trees have not the expected shape.")
            left_head = head.children[9]
            left_tail = head.children[10]
            right_term = tail.children[6]

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


    def specification(self):
        labels = DataGroup("para", self.labels)
        para_labels = self.Para(self.labels, self.dimensions)
        paratuples = self.ParaTuples(para_labels, max_length=max(self.dimensions))
        paratupletuples = self.ParaTupleTuples(paratuples)
        dimension = DataGroup("dimension", self.dimensions)

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
            .parameter("para", para_labels, lambda v: [(("swap", 0, v["io"]), v["io"], v["io"]),
                                                       (("swap", 0, None), v["io"], v["io"]),
                                                       (("swap", None, v["io"]), v["io"], v["io"]),
                                                       (("swap", None, None), v["io"], v["io"]),
                                                       (("swap", 0, v["io"]), v["io"], None),
                                                       (("swap", 0, None), v["io"], None),
                                                       (("swap", None, v["io"]), v["io"], None),
                                                       (("swap", None, None), v["io"], None),
                                                       (("swap", 0, v["io"]), None, v["io"]),
                                                       (("swap", 0, None), None, v["io"]),
                                                       (("swap", None, v["io"]), None, v["io"]),
                                                       (("swap", None, None), None, v["io"]),
                                                       (("swap", 0, v["io"]), None, None),
                                                       (("swap", 0, None), None, None),
                                                       (("swap", None, v["io"]), None, None),
                                                       (("swap", None, None), None, None), None])
                                                        # (None, None, None)])
            .suffix(Constructor("DAG_component", Constructor("input", Var("io"))
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
            .parameter("para", para_labels, lambda v: [(("swap", v["n"], v["m"]), v["io"], v["io"]),
                                                       (("swap", v["n"], None), v["io"], v["io"]),
                                                       (("swap", None, v["m"]), v["io"], v["io"]),
                                                       (("swap", None, None), v["io"], v["io"]),
                                                       (("swap", v["n"], v["m"]), v["io"], None),
                                                       (("swap", v["n"], None), v["io"], None),
                                                       (("swap", None, v["m"]), v["io"], None),
                                                       (("swap", None, None), v["io"], None),
                                                       (("swap", v["n"], v["m"]), None, v["io"]),
                                                       (("swap", v["n"], None), None, v["io"]),
                                                       (("swap", None, v["m"]), None, v["io"]),
                                                       (("swap", None, None), None, v["io"]),
                                                       (("swap", v["n"], v["m"]), None, None),
                                                       (("swap", v["n"], None), None, None),
                                                       (("swap", None, v["m"]), None, None),
                                                       (("swap", None, None), None, None), None])
                                                       #(None, None, None)])
            .suffix(Constructor("DAG_component", Constructor("input", Var("io"))
                                & Constructor("input", Literal(None))
                                & Constructor("output", Var("io"))
                                & Constructor("output", Literal(None))
                                & Constructor("structure", Var("para"))
                                & Constructor("structure", Literal(None))
                                ) & Constructor("non_ID")
                    ),
            "node": SpecificationBuilder()
            .parameter("l", labels)
            .parameter("i", dimension)
            .parameter("o", dimension)
            .parameter("para", para_labels, lambda v: [(v["l"], v["i"], v["o"]),
                                                       (v["l"], v["i"], None),
                                                       (v["l"], None, v["o"]),
                                                       (None, v["i"], v["o"]),
                                                       (v["l"], None, None),
                                                       (None, None, v["o"]),
                                                       (None, v["i"], None), None])
                                                       #(None, None, None)])
            .suffix(Constructor("DAG_component",
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
                                & Constructor("structure", Literal(None))
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
                                & Constructor("structure", Literal(None))
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
                                  & Constructor("structure", Literal(None))
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
                                  & Constructor("structure", Literal(None))
                                  )
                      & Constructor("non_ID") & Constructor("last", Constructor("non_ID")))
                     )
                    ),

            "before_singleton": SpecificationBuilder()
            .parameter("i", dimension)
            .parameter("o", dimension)
            .parameter("ls", paratupletuples)
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
                                & Constructor("structure", Var("ls"))
                                )),

            "before_cons": SpecificationBuilder()
            .parameter("i", dimension)
            .parameter("j", dimension)
            .parameter("o", dimension)
            .parameter("ls", paratupletuples)
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
                                & Constructor("structure", Var("ls")))),
        }

    # Interpretations of terms are algebras in my language

    def pretty_term_algebra(self):
        return {
            "edges": (lambda io, para: f"edges({io})"),

            "swap": (lambda io, n, m, para: f"swap({io}, {n}, {m})"),

            "node": (lambda l, i, o, para: f"node({l}, {i}, {o})"),

            "beside_singleton": (lambda i, o, ls, para, x: f"{x})"),

            "beside_cons": (lambda i, i1, i2, o, o1, o2, ls, head, tail, x, y: f"{x} || {y}"),

            "before_singleton": (lambda i, o, ls, ls1, x: f"({x}"),

            "before_cons": (lambda i, j, o, ls, head, tail, x, y: f"({x} ; {y}"),
        }

    def unique_id(self):
        id = self.id_seed
        self.id_seed += 1
        return id


    def path_algebra(self):
        return {
            "edges": (lambda io, para: lambda id, inputs: ([], inputs)),

            "swap": (lambda io, n, m, para: lambda id, inputs: ([], inputs[n:] + inputs[:n])),

            "node": (lambda l, i, o, para: lambda id, inputs: ([(x,l + str(id)) for x in inputs],  [l + str(id) for _ in range(0,o)])),

            "beside_singleton": (lambda i, o, ls, para, x: x),

            "beside_cons": (lambda i, i1, i2, o, o1, o2, ls, head, tail, x, y: lambda id, inputs: (x(id, inputs[:i1])[0] + y((id[0], id[1] + 1), inputs[i1:])[0], x(id, inputs[:i1])[1] + y((id[0], id[1] + 1), inputs[i1:])[1])),

            "before_singleton": (lambda i, o, ls, ls1, x: x),

            "before_cons": (lambda i, j, o, ls, head, tail, x, y: lambda id, inputs: (y((id[0] + 1, id[1]), x(id, inputs)[1])[0] + x(id, inputs)[0], y((id[0] + 1, id[1]), x(id, inputs)[1])[1])),
        }
