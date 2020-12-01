import numpy as np
import os
import argparse

"""
Use a recursive :( strategy to find the CNF of a one-hot encoding with N variables
It will work well with even 20 variables without being to heavy on the CPU
"""

def find_CNF(value, number, relation, base):
    """
    relation (lt, gt, eq, ne) value (0, 1, 2, ...) variable out of number (4, 5, 6, ...) are true
    """
    starting_vector = []
    clauses = []
    _find_CNF(value, number, relation, starting_vector, clauses, base)
    return clauses

def _find_CNF(value, number, relation, vector, clauses, base):
    if len(vector) == number:
        summed = sum(vector)
        if relation == 'lt' and summed >= value or \
            relation == 'gt' and summed <= value or \
            relation == 'eq' and summed != value or \
            relation == 'ne' and summed == value:
                clauses.append("Or({})".format(
                    ", ".join("~X{}{}".format(base, i) if x else "X{}{}".format(base, i) for i, x in enumerate(vector))
                ))
    else:
        _find_CNF(value, number, relation, vector + [0], clauses, base)
        _find_CNF(value, number, relation, vector + [1], clauses, base)


def find_DNF(value, number, relation, base):
    """
    relation (lt, gt, eq, ne) value (0, 1, 2, ...) variable out of number (4, 5, 6, ...) are true
    """
    starting_vector = []
    clauses = []
    _find_DNF(value, number, relation, starting_vector, clauses, base)
    res = "Or({})".format(", ".join(clauses))
    return [res]

def _find_DNF(value, number, relation, vector, clauses, base):
    if len(vector) == number:
        summed = sum(vector)
        if relation == 'lt' and summed < value or \
            relation == 'gt' and summed > value or \
            relation == 'eq' and summed == value or \
            relation == 'ne' and summed != value:
                clauses.append("And({})".format(
                    ", ".join("X{}{}".format(base, i) if x else "~X{}{}".format(base, i) for i, x in enumerate(vector))
                ))
    else:
        _find_DNF(value, number, relation, vector + [0], clauses, base)
        _find_DNF(value, number, relation, vector + [1], clauses, base)



# main module
if __name__ == "__main__":
    """
    This script creates the CNF formula to express contraints like:
    - at least v variables out of n are true
    - at most v variables out of n are true
    - exactly v variables out of n are true
    - exactly not v variables out of n are true
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--relation", type=str, required=True,
                        help="Relation on which the constraint will be builed. Should be one of lt (less than), gt (greater than), eq (equal), ne (not equal)")
    parser.add_argument("-v", "--value", type=int, required=True,
                        help="The target value on which the constraint will be builded. Should be an integer")
    parser.add_argument("-n", "--number", type=int, required=True,
                        help="The number of variables to use in the constraints")
    parser.add_argument("-b", "--base", type=str, required=False, default="",
                        help="Base address of each variable. Ex 0.0. means that each variable will be like X0.0.V...")
    parser.add_argument("-o", "--output_file", type=str, required=True,
                        help="Output sympy file to which the CNF will be written")
    parser.add_argument("--mode", type=str, choices=["cnf", "dnf"], default="cnf",
                        help="Do you want a CNF or a DNF?")

    args = parser.parse_args()

    variable_number = args.number
    variable_value = args.value
    relation = args.relation
    base = args.base
    mode = args.mode

    # get paths of samples
    output_file = args.output_file
    print("Will save computed CNF to %s" % output_file)
    # eventually clean and re-create folder

    print("Computing CNF formulae")
    if mode == "cnf":
        res = find_CNF(variable_value, variable_number, relation, base)
    else:
        res = find_DNF(variable_value, variable_number, relation, base)

    print("Saving to file in sympy format")
    with open(output_file, 'w') as ofile:
        ofile.write("# File generated automatically with count_to_cnf.py tool. Use with caution\n\n")
        ofile.write("shape [%d]\n" % variable_number)
        for row in res:
            ofile.write(row + "\n")

    print("Done!")
