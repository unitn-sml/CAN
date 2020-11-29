import itertools

"""
Script to create still life constraints in sympy syntax to be later used for the semantic loss.
"""


def corner_constraints(n):
    """
    Constraints for still life corners written in sympy logic, the first variable
    is to be considered the corner.

    :return:
    """
    res = []
    ifzero = "Implies(~X.%s.%s, Nand(X.%s.%s, X.%s.%s, X.%s.%s))"
    ifone = "Implies(X.%s.%s, Or(And(X.%s.%s, X.%s.%s, X.%s.%s), And(~X.%s.%s, X.%s.%s, X.%s.%s), And(X.%s.%s, " \
            "~X.%s.%s, X.%s.%s), And(X.%s.%s, X.%s.%s, ~X.%s.%s))) "

    # top left corner
    fillerzero = (0, 0, 0, 1, 1, 1, 1, 0)
    fillerone = (0, 0) + (0, 1, 1, 1, 1, 0) * 4
    res.append(ifzero % fillerzero)
    res.append(ifone % fillerone)

    # top right corner
    fillerzero = (0, n - 1, 1, n - 1, 1, n - 2, 0, n - 2)
    fillerone = (0, n - 1) + (1, n - 1, 1, n - 2, 0, n - 2) * 4
    res.append(ifzero % fillerzero)
    res.append(ifone % fillerone)

    # bottomleft corner
    fillerzero = (n - 1, 0, n - 2, 0, n - 2, 1, n - 1, 1)
    fillerone = (n - 1, 0) + (n - 2, 0, n - 2, 1, n - 1, 1) * 4
    res.append(ifzero % fillerzero)
    res.append(ifone % fillerone)

    # bottomright corner
    fillerzero = (n - 1, n - 1, n - 1, n - 2, n - 2, n - 2, n - 2, n - 1)
    fillerone = (n - 1, n - 1) + (n - 1, n - 2, n - 2, n - 2, n - 2, n - 1) * 4
    res.append(ifzero % fillerzero)
    res.append(ifone % fillerone)

    return "\n".join(res)


def n_k_combinations(n, k, indices):
    """
    Possible combination of n variables, with k of them being 1 and n-k 0.
    :param n:
    :return:
    """

    # indexes of k variables that are set to 1
    k_indexes = list(itertools.permutations(range(n), k))
    k_indexes = [tuple(sorted(el)) for el in k_indexes]
    k_indexes = set(k_indexes)
    possible_combinations = []

    # for each K-tuple of indexes that should be set to 1, create an assignment
    variables = "And(%s)"
    variables = variables % ",".join(["%sX.%s.%s" % ("%s", pair[0], pair[1]) for pair in indices])

    for indexes in sorted(k_indexes):
        # all values start as 0, 3 of them are set to 1
        values = ["~" for _ in range(n)]
        for index in indexes:
            values[index] = ""
        possible_combinations.append(variables % tuple(values))
    possible_combinations = ",".join(possible_combinations)
    return possible_combinations


def edge_constraints(N):
    """
    Constraints for still life edges written in sympy logic, the first variable
    is to be considered the edge, the rest (5) are the other ones, for a total of six variables.

    :return:
    """
    res = []

    # if 0 then there cannot be 3 neighbours which are 1
    # Not(Or(all combinations of 3 variables to 1 and rest to 0))
    edges = []
    edges.extend([(0, i) for i in range(1, N - 1)])
    edges.extend([(i, N - 1) for i in range(1, N - 1)])
    edges.extend([(N - 1, i) for i in range(1, N - 1)])
    edges.extend([(i, 0) for i in range(1, N - 1)])
    for x, y in edges:
        # top edge
        if x == 0:
            indices = [(x, y + 1), (x + 1, y + 1), (x + 1, y), (x + 1, y - 1), (x, y - 1)]
        # right edge
        if y == (N - 1):
            indices = [(x + 1, y), (x + 1, y - 1), (x, y - 1), (x - 1, y - 1), (x - 1, y)]
        # bottom edge
        if x == (N - 1):
            indices = [(x, y + 1), (x, y - 1), (x - 1, y - 1), (x - 1, y), (x - 1, y + 1)]
        # left edge
        if y == 0:
            indices = [(x - 1, y), (x - 1, y + 1), (x, y + 1), (x + 1, y + 1), (x + 1, y)]

        ifzero = "Implies(~X.%s.%s, Nor(%s))" % (x, y, "%s")
        ifzero = ifzero % n_k_combinations(5, 3, indices=indices)
        res.append(ifzero)

        ifone = "Implies(X.%s.%s, Nor(%s))" % (x, y, "%s")
        excluded = [0, 1, 4, 5]
        ifone = ifone % ",".join([n_k_combinations(5, i, indices=indices) for i in excluded])
        res.append(ifone)

    return "\n".join(res)


def internal_constraints(N):
    """
    Constraints for still life internal points written in sympy logic, the first variable
    is to be considered the center, the rest (8) are the other ones, for a total of 9 variables.

    :return:
    """
    res = []
    for x in range(1, N - 1):
        for y in range(1, N - 1):
            indices = [(x - 1, y - 1), (x - 1, y), (x - 1, y + 1), (x, y + 1), (x + 1, y + 1), (x + 1, y),
                       (x + 1, y - 1), (x, y - 1)]

            ifzero = "Implies(~X.%s.%s, Nor(%s))" % (x, y, "%s")
            ifzero = ifzero % n_k_combinations(8, 3, indices=indices)
            res.append(ifzero)

            ifone = "Implies(X.%s.%s, Nor(%s))" % (x, y, "%s")
            excluded = [0, 1, 4, 5, 6, 7, 8]
            ifone = ifone % ",".join([n_k_combinations(8, i, indices=indices) for i in excluded])
            res.append(ifone)

    return "\n".join(res)


N = 28
corners = corner_constraints(N)
edges = edge_constraints(N)
internals = internal_constraints(N)

with open("still_life_full.txt", "w") as file:
    file.write("shape [28,28]\n")
    file.write(corners)
    file.write("\n")
    file.write(edges)
    file.write("\n")
    file.write(internals)

