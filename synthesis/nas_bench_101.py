from cl3s import (SpecificationBuilder, Constructor, Literal, Var, Group, DataGroup,
                  DerivationTree, SearchSpaceSynthesizer)
from typing import Any



class NASBench101_Repo:
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

    def __init__(self, labels):
        # additionally to labeled nodes, we have (unlabelled) edges, that needs to be handled additionally
        if "swap" in labels:
            raise ValueError("Label 'swap' is reserved and cannot be used as a node label.")
        self.labels = labels
        self.dimensions = list(range(1, 10)) # NAS-Bench-101 contains at most 9 edges

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
            if len(head.children) != 5 or len(tail.children) != 11:
                raise ValueError("Derivation trees have not the expected shape.")
            left_term = head.children[4]
            right_head = tail.children[9]
            right_tail = tail.children[10]

            left_term_root = left_term.root
            right_head_root = right_head.root
            right_tail_root = right_tail.root
            if (left_term_root == "swap") and "beside_cons" in right_head_root and "before_singleton" in right_tail_root:
                if len(left_term.children) != 4 or len(right_head.children) != 14 or len(right_tail.children) != 5:
                    raise ValueError("Derivation trees have not the expected shape.")
                m = left_term.children[1]
                n = left_term.children[2]
                x_n = right_head.children[1]
                x_p = right_head.children[4]
                right_head_tail = right_head.children[13]
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
                if len(left_term.children) != 4 or len(right_head.children) != 14 or len(right_tail.children) != 11:
                    raise ValueError("Derivation trees have not the expected shape.")
                m = left_term.children[1]
                n = left_term.children[2]
                x_n = right_head.children[1]
                x_p = right_head.children[4]
                right_head_tail = right_head.children[13]
                right_tail_head = right_tail.children[9]
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
            if len(head.children) != 14 or len(tail.children) != 5:
                raise ValueError("Derivation trees have not the expected shape.")
            left_head = head.children[12]
            left_tail = head.children[13]
            right_term = tail.children[4]

            left_swap = left_head.root
            left_tail_root = left_tail.root
            right_term_root = right_term.root
            if (left_swap == "swap") and "beside_singleton" in left_tail_root and "beside_cons" in right_term_root:
                if len(left_head.children) != 4 or len(left_tail.children) != 5 or len(right_term.children) != 14:
                    raise ValueError("Derivation trees have not the expected shape.")
                m = left_head.children[1]
                n = left_head.children[2]
                left_tail_term = left_tail.children[4]  # swap(p, 0, p)
                right_head = right_term.children[12]  # swap(n, 0, n)
                right_tail = right_term.children[13]

                left_tail_swap = left_tail_term.root
                right_head_root = right_head.root
                right_tail_root = right_tail.root
                if left_tail_swap == "edges" and right_head_root == "edges" and "beside_singleton" in right_tail_root:
                    if len(left_tail_term.children) != 2 or len(right_head.children) != 2 or len(right_tail.children) != 5:
                        raise ValueError("Derivation trees have not the expected shape.")
                    p = left_tail_term.children[0]
                    right_n = right_head.children[0]

                    right_tail_term = right_tail.children[4]  # swap(m+p, m, p)
                    right_tail_term_root = right_tail_term.root
                    if (right_tail_term_root == "swap") and n == right_n:
                        if len(right_tail_term.children) != 4:
                            raise ValueError("Derivation trees have not the expected shape.")
                        right_m = right_tail_term.children[1]
                        right_p = right_tail_term.children[2]
                        if m == right_m and p == right_p:
                            return False
        elif "beside_cons" in left and "before_cons" in right:
            if len(head.children) != 14 or len(tail.children) != 11:
                raise ValueError("Derivation trees have not the expected shape.")
            left_head = head.children[12]
            left_tail = head.children[13]
            right_term = tail.children[9]

            left_swap = left_head.root
            left_tail_root = left_tail.root
            right_term_root = right_term.root
            if (left_swap == "swap") and "beside_singleton" in left_tail_root and "beside_cons" in right_term_root:
                if len(left_head.children) != 4 or len(left_tail.children) != 5 or len(right_term.children) != 14:
                    raise ValueError("Derivation trees have not the expected shape.")
                m = left_head.children[1]
                n = left_head.children[2]
                left_tail_term = left_tail.children[4]  # swap(p, 0, p)
                right_head = right_term.children[12]  # swap(n, 0, n)
                right_tail = right_term.children[13]

                left_tail_swap = left_tail_term.root
                right_head_root = right_head.root
                right_tail_root = right_tail.root
                if left_tail_swap == "edges" and right_head_root == "edges" and "beside_singleton" in right_tail_root:
                    if len(left_tail_term.children) != 2 or len(right_head.children) != 2 or len(right_tail.children) != 5:
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
                if right_beside_root == "swap":
                    if len(right_beside.children) != 4:
                        raise ValueError("Derivation trees have not the expected shape.")
                    right_n = right_beside.children[1]
                    right_m = right_beside.children[2]
                    if m == right_m and n == right_n:
                        return False
        elif "beside_singleton" in left and "before_cons" in right:
            if len(head.children) != 5 or len(tail.children) != 11:
                raise ValueError("Derivation trees have not the expected shape.")
            left_term = head.children[4]
            right_term = tail.children[9]

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
            if len(head.children) != 14 or len(tail.children) != 5:
                raise ValueError("Derivation trees have not the expected shape.")
            left_head = head.children[12]
            left_tail = head.children[13]
            right_term = tail.children[4]

            left_head_root = left_head.root
            left_tail_root = left_tail.root
            right_term_root = right_term.root
            if left_head_root == "edges" and "beside_singleton" in left_tail_root and "beside_cons" in right_term_root:
                if len(left_head.children) != 2 or len(left_tail.children) != 5 or len(right_term.children) != 14:
                    raise ValueError("Derivation trees have not the expected shape.")
                m = left_head.children[0]

                left_tail_term = left_tail.children[4]
                right_head = right_term.children[12]
                right_tail = right_term.children[13]

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
            if len(head.children) != 14 or len(tail.children) != 11:
                raise ValueError("Derivation trees have not the expected shape.")
            left_head = head.children[12]
            left_tail = head.children[13]
            right_term = tail.children[9]

            left_head_root = left_head.root
            left_tail_root = left_tail.root
            right_term_root = right_term.root
            if left_head_root == "edges" and "beside_singleton" in left_tail_root and "beside_cons" in right_term_root:
                if len(left_head.children) != 2 or len(left_tail.children) != 5 or len(right_term.children) != 14:
                    raise ValueError("Derivation trees have not the expected shape.")
                m = left_head.children[0]

                left_tail_term = left_tail.children[4]
                right_head = right_term.children[12]
                right_tail = right_term.children[13]

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
        labels = DataGroup("para", self.labels)
        dimension = DataGroup("dimension", self.dimensions)
        number = DataGroup("number", self.dimensions + [0])

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
            .parameter("e", number, lambda v: [v["io"]])
            .suffix(Constructor("DAG_component", Constructor("input", Var("io"))
                                & Constructor("input", Literal(None))
                                & Constructor("output", Var("io"))
                                & Constructor("output", Literal(None))
                                & Constructor("number_of_edges", Var("e"))
                                & Constructor("number_of_nodes", Literal(0))
                                ) & Constructor("ID")
                    ),

            "swap": SpecificationBuilder()
            .parameter("io", dimension)
            .parameter("n", dimension, lambda v: range(1, v["io"]))
            .parameter("m", dimension, lambda v: [v["io"] - v["n"]]) # m > 0
            .parameter("e", number, lambda v: [v["io"]])
            .suffix(Constructor("DAG_component", Constructor("input", Var("io"))
                                & Constructor("input", Literal(None))
                                & Constructor("output", Var("io"))
                                & Constructor("output", Literal(None))
                                & Constructor("number_of_edges", Var("e"))
                                & Constructor("number_of_nodes", Literal(0))
                                ) & Constructor("non_ID")
                    ),
            "node": SpecificationBuilder()
            .parameter("l", labels)
            .parameter("i", dimension)
            .parameter("o", dimension)
            .parameter("e", number, lambda v: [v["i"] + v["o"]])
            .suffix(Constructor("DAG_component",
                                Constructor("input", Var("i"))
                                & Constructor("input", Literal(None))
                                & Constructor("output", Var("o"))
                                & Constructor("output", Literal(None))
                                & Constructor("number_of_edges", Var("e"))
                                & Constructor("number_of_nodes", Literal(1))
                                ) & Constructor("non_ID")
                    ),
            #'''
            "beside_singleton": SpecificationBuilder()
            .parameter("i", dimension)
            .parameter("o", dimension)
            .parameter("e", number)
            .parameter_constraint(lambda v: v["i"] <= v["e"] <= (v["i"] + v["o"]))
            .parameter("n", number)
            .suffix(
                ((Constructor("DAG_component",
                              Constructor("input", Var("i"))
                              & Constructor("output", Var("o"))
                              & Constructor("number_of_edges", Var("e"))
                              & Constructor("number_of_nodes", Var("n"))
                              )
                     & Constructor("non_ID")
                  )
                 **
                 (Constructor("DAG_parallel",
                              Constructor("input", Var("i"))
                              & Constructor("input", Literal(None))
                              & Constructor("output", Var("o"))
                              & Constructor("output", Literal(None))
                              & Constructor("number_of_edges", Var("e"))
                              & Constructor("number_of_nodes", Var("n"))
                              )
                  & Constructor("non_ID") & Constructor("last", Constructor("non_ID"))
                  )
                 )
                &
                ((Constructor("DAG_component",
                              Constructor("input", Var("i"))
                              & Constructor("output", Var("o"))
                              & Constructor("number_of_edges", Var("e"))
                              & Constructor("number_of_nodes", Literal(1))
                              )
                     & Constructor("ID")
                     )
                 **
                 (Constructor("DAG_parallel",
                              Constructor("input", Var("i"))
                              & Constructor("input", Literal(None))
                              & Constructor("output", Var("o"))
                              & Constructor("output", Literal(None))
                              & Constructor("number_of_edges", Var("e"))
                              & Constructor("number_of_nodes", Literal(1))
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
            .parameter("e", number)
            .parameter_constraint(lambda v: v["i"] <= v["e"] <= (v["i"] + v["o"]))
            .parameter("e1", number)
            .parameter_constraint(lambda v: v["i1"] <= v["e1"] <= (v["i1"] + v["o1"]))
            .parameter("e2", number, lambda v: [v["e"] - v["e1"]])
            .parameter("n", number)
            .parameter("n1", number)
            .parameter("n2", number, lambda v: [v["n"] - v["n1"]])
            .suffix(
                    ((Constructor("DAG_component",
                                  Constructor("input", Var("i1"))
                                  & Constructor("output", Var("o1"))
                                  & Constructor("number_of_edges", Var("e1"))
                                  & Constructor("number_of_nodes", Var("n1"))
                                  )
                      & Constructor("ID"))
                    **
                    (Constructor("DAG_parallel",
                                 Constructor("input", Var("i2"))
                                 & Constructor("output", Var("o2"))
                                 & Constructor("number_of_edges", Var("e2"))
                                 & Constructor("number_of_nodes", Var("n2"))
                                 )
                     & Constructor("non_ID") & Constructor("last", Constructor("non_ID")))
                     **
                     (Constructor("DAG_parallel",
                                  Constructor("input", Var("i"))
                                  & Constructor("input", Literal(None))
                                  & Constructor("output", Var("o"))
                                  & Constructor("output", Literal(None))
                                  & Constructor("number_of_edges", Var("e"))
                                  & Constructor("number_of_nodes", Var("n"))
                                )
                      & Constructor("non_ID") & Constructor("last", Constructor("ID")))
                     )
                    &
                    ((Constructor("DAG_component",
                                  Constructor("input", Var("i1"))
                                  & Constructor("output", Var("o1"))
                                  & Constructor("number_of_edges", Var("e1"))
                                  & Constructor("number_of_nodes", Var("n1"))
                                  )
                      & Constructor("non_ID"))
                     **
                     (Constructor("DAG_parallel",
                                  Constructor("input", Var("i2"))
                                  & Constructor("output", Var("o2"))
                                  & Constructor("number_of_edges", Var("e2"))
                                  & Constructor("number_of_nodes", Var("n2"))
                                  )
                      & Constructor("ID"))
                     **
                     (Constructor("DAG_parallel",
                                  Constructor("input", Var("i"))
                                  & Constructor("input", Literal(None))
                                  & Constructor("output", Var("o"))
                                  & Constructor("output", Literal(None))
                                  & Constructor("number_of_edges", Var("e"))
                                  & Constructor("number_of_nodes", Var("n"))
                                  )
                      & Constructor("non_ID") & Constructor("last", Constructor("non_ID")))
                     )
                    &
                    ((Constructor("DAG_component",
                                  Constructor("input", Var("i1"))
                                  & Constructor("output", Var("o1"))
                                  & Constructor("number_of_edges", Var("e1"))
                                  & Constructor("number_of_nodes", Var("n1"))
                                  )
                      & Constructor("non_ID"))
                     **
                     (Constructor("DAG_parallel",
                                  Constructor("input", Var("i2"))
                                  & Constructor("output", Var("o2"))
                                  & Constructor("number_of_edges", Var("e2"))
                                  & Constructor("number_of_nodes", Var("n2"))
                                  )
                      & Constructor("non_ID"))
                     **
                     (Constructor("DAG_parallel",
                                  Constructor("input", Var("i"))
                                  & Constructor("input", Literal(None))
                                  & Constructor("output", Var("o"))
                                  & Constructor("output", Literal(None))
                                  & Constructor("number_of_edges", Var("e"))
                                  & Constructor("number_of_nodes", Var("n"))
                                  )
                      & Constructor("non_ID") & Constructor("last", Constructor("non_ID")))
                     )
                    ),

            "before_singleton": SpecificationBuilder()
            .parameter("i", dimension)
            .parameter("o", dimension)
            .parameter("e", number)
            .parameter_constraint(lambda v: v["i"] <= v["e"] <= (v["i"] + v["o"]))
            .parameter("n", number)
            .argument("x", Constructor("DAG_parallel",
                                       Constructor("input", Var("i"))
                                       & Constructor("output", Var("o"))
                                       & Constructor("number_of_edges", Var("e"))
                                       & Constructor("number_of_nodes", Var("n"))
                                       ) & Constructor("non_ID"))
            .suffix(Constructor("DAG",
                                Constructor("input", Var("i"))
                                & Constructor("input", Literal(None))
                                & Constructor("output", Var("o"))
                                & Constructor("output", Literal(None))
                                & Constructor("number_of_edges", Var("e"))
                                & Constructor("number_of_nodes", Var("n"))
                                )),

            "before_cons": SpecificationBuilder()
            .parameter("i", dimension)
            .parameter("j", dimension)
            .parameter("o", dimension)
            .parameter("e", number)
            .parameter_constraint(lambda v: v["i"] <= v["e"] <= (v["i"] + v["o"]))
            .parameter("e1", number)
            .parameter_constraint(lambda v: v["i"] <= v["e1"] <= (v["i"] + v["j"]))
            .parameter("e2", number, lambda v: [v["e"] - v["e1"] + v["j"]])
            .parameter("n", number)
            .parameter("n1", number)
            .parameter("n2", number, lambda v: [v["n"] - v["n1"]])
            .argument("x", Constructor("DAG_parallel",
                                       Constructor("input", Var("i"))
                                       & Constructor("output", Var("j"))
                                       & Constructor("number_of_edges", Var("e1"))
                                       & Constructor("number_of_nodes", Var("n1"))
                                       ) & Constructor("non_ID"))
            .argument("y", Constructor("DAG",
                                       Constructor("input", Var("j"))
                                       & Constructor("output", Var("o"))
                                       & Constructor("number_of_edges", Var("e2"))
                                       & Constructor("number_of_nodes", Var("n2"))
                                       ))
            .constraint(lambda v: self.swaplaw1(v["x"], v["y"]))
            .constraint(lambda v: self.swaplaw2(v["x"], v["y"]))
            .constraint(lambda v: self.swaplaw3(v["x"], v["y"]))
            .constraint(lambda v: self.swaplaw4(v["x"], v["y"]))
            .constraint(lambda v: self.dag_constraint(v["x"], v["y"], v["i"])) # DAG: At most one edge between any two nodes
            .suffix(Constructor("DAG",
                                Constructor("input", Var("i"))
                                & Constructor("input", Literal(None))
                                & Constructor("output", Var("o"))
                                & Constructor("output", Literal(None))
                                & Constructor("number_of_edges", Var("e"))
                                & Constructor("number_of_nodes", Var("n"))
                                )),
        }

    # Interpretations of terms are algebras in my language

    def pretty_term_algebra(self):
        return {
            "edges": (lambda io, e: f"edges({io})"),

            "swap": (lambda io, n, m, e: f"swap({io}, {n}, {m})"),

            "node": (lambda l, i, o, e: f"node({l}, {i}, {o})"),

            "beside_singleton": (lambda i, o, e, n, x: f"{x})"),

            "beside_cons": (lambda i, i1, i2, o, o1, o2, e, e1, e2, n, n1, n2, x, y: f"{x} || {y}"),

            "before_singleton": (lambda i, o, e, n, x: f"({x}"),

            "before_cons": (lambda i, j, o, e, e1, e2, n, n1, n2, x, y: f"({x} ; {y}"),
        }

    def edgelist_algebra(self):
        return {
            "edges": (lambda io, e: lambda id, inputs: ([], inputs, {})),

            "swap": (lambda io, n, m, e: lambda id, inputs: ([], inputs[n:] + inputs[:n], {})),

            "node": (lambda l, i, o, e: lambda id, inputs: ([(x,str((l, i, o)) + str(id)) for x in inputs],  [str((l, i, o)) + str(id) for _ in range(0,o)], {str((l, i, o)) + str(id) : id})),

            "beside_singleton": (lambda i, o, e, n, x: x),

            "beside_cons": (lambda i, i1, i2, o, o1, o2, e, e1, e2, n, n1, n2, x, y: lambda id, inputs:
                (x(id, inputs[:i1])[0] + y((id[0], id[1] + 0.2), inputs[i1:])[0],
                 x(id, inputs[:i1])[1] + y((id[0], id[1] + 0.2), inputs[i1:])[1],
                 x(id, inputs[:i1])[2] | y((id[0], id[1] + 0.2), inputs[i1:])[2])),

            "before_singleton": (lambda i, o, e, n, x: (x, i)),

            "before_cons": (lambda i, j, o, e, e1, e2, n, n1, n2, x, y: (lambda id, inputs:
                                                                      (y[0]((id[0] + 2.5, id[1]), x(id, inputs)[1])[0] + x(id, inputs)[0],
                                                                       y[0]((id[0] + 2.5, id[1]), x(id, inputs)[1])[1],
                                                                       y[0]((id[0] + 2.5, id[1]), x(id, inputs)[1])[2] | x(id, inputs)[2]),
                                                                      i)),
        }


if __name__ == "__main__":
    repo = NASBench101_Repo(["3x3Conv", "1x1Conv", "3x3MaxPool"])

    target0 = Constructor("DAG",
                         Constructor("input", Literal(None))
                         & Constructor("output", Literal(None))
                         & Constructor("number_of_edges", Literal(3))
                         & Constructor("number_of_nodes", Literal(1))
                         )

    target_parallel = Constructor("DAG_parallel",
                         Constructor("input", Literal(2))
                         & Constructor("output", Literal(1))
                         & Constructor("number_of_edges", Literal(3))
                         & Constructor("number_of_nodes", Literal(1))
                         )

    target = target0

    print("start synthesis")
    synthesizer = SearchSpaceSynthesizer(repo.specification(), {})
    search_space = synthesizer.construct_search_space(target).prune()
    print("finish synthesis, start sampling")

    terms = search_space.enumerate_trees(target, 100)

    terms = list(terms)

    print(len(terms))

    for t in terms:
        print(t.interpret(repo.pretty_term_algebra()))