import itertools

"""
Script to create still life constraints in sympy syntax to be later used for the semantic loss.
"""


def corner_constraints():
    """
    Constraints for still life corners written in sympy logic, the first variable
    is to be considered the corner.

    :return:
    """
    res = []

    # if 0 then not all 3 neighbours can be 1
    ifzero = "Implies(~X0, Nand(X1, X2, X3))"
    res.append(ifzero)

    # if 1 then either all 3 neighbours or 2 of them are 1
    ifone = "Implies(X0, Or(And(X1, X2, X3), And(~X1, X2, X3), And(X1, ~X2, X3), And(X1, X2, ~X3)))"
    res.append(ifone)

    return "\n".join(res)


def n_k_combinations(n, k):
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
    variables = variables % ",".join(["%sX%s" % ("%s", i) for i in range(1, n + 1)])

    for indexes in sorted(k_indexes):
        # all values start as 0, 3 of them are set to 1
        values = ["~" for _ in range(n)]
        for index in indexes:
            values[index] = ""
        possible_combinations.append(variables % tuple(values))
    possible_combinations = ",".join(possible_combinations)
    return possible_combinations


def edge_constraints():
    """
    Constraints for still life edges written in sympy logic, the first variable
    is to be considered the edge, the rest (5) are the other ones, for a total of six variables.

    :return:
    """
    res = []

    # if 0 then there cannot be 3 neighbours which are 1
    # Not(Or(all combinations of 3 variables to 1 and rest to 0))

    ifzero = "Implies(~X0, Nor(%s))"
    ifzero = ifzero % n_k_combinations(5, 3)
    res.append(ifzero)

    """
    old version
    ifone = "Implies(X0, Or(%s))"
    two_active = n_k_combinations(5, 2)
    three_active = n_k_combinations(5, 3)
    ifone = ifone % ",".join([two_active, three_active])
    """

    ifone = "Implies(X0, Nor(%s))"
    excluded = [0, 1, 4, 5]
    ifone = ifone % ",".join([n_k_combinations(5, i) for i in excluded])

    res.append(ifone)

    return "\n".join(res)


def internal_constraints():
    """
    Constraints for still life internal points written in sympy logic, the first variable
    is to be considered the center, the rest (8) are the other ones, for a total of 9 variables.

    :return:
    """
    res = []

    # if 0 then there cannot be 3 neighbours which are 1
    # Not(Or(all combinations of 3 variables to 1 and rest to 0))
    ifzero = "Implies(~X0, Nor(%s))"
    ifzero = ifzero % n_k_combinations(8, 3)
    res.append(ifzero)

    """
    old version
    ifone = "Implies(X0, Or(%s))"
    two_active = n_k_combinations(8, 2)
    three_active = n_k_combinations(8, 3)
    ifone = ifone % ",".join([two_active, three_active])

    """

    ifone = "Implies(X0, Nor(%s))"
    excluded = [0, 1, 4, 5, 6, 7, 8]
    ifone = ifone % ",".join([n_k_combinations(8, i) for i in excluded])
    res.append(ifone)

    return "\n".join(res)


corners = corner_constraints()
edges = edge_constraints()
internals = internal_constraints()

with open("still_life_corners.txt", "w") as file:
    file.write("shape [4]\n")
    file.write(corners)

with open("still_life_edges.txt", "w") as file:
    file.write("shape [6]\n")
    file.write(edges)

with open("still_life_internals.txt", "w") as file:
    file.write("shape [9]\n")
    file.write(internals)
