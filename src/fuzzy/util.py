"""
Module containing utility stuff for the fuzzy_logic_loss directory.
"""
import os
from os import listdir
from os.path import isfile, join


def _get_constraints_names():
    """
    Get the names of all constraints for which there is a .fuzzy file in
    in/semantic_loss_constraints/constraints_as_sympy_tree.
    Expect the cwd to be the /src directory of the repository.
    :return: List of strings indicating names of constraints that can be instantiated.
    """
    cwd = os.getcwd()
    assert cwd[-4:] == "/src", "Expected to be in the src directory of the repository, the cwd is instead {}".format(cwd)

    filespath = os.path.join("in", "semantic_loss_constraints" , "constraints_as_sympy_tree")
    files = sorted([f for f in listdir(filespath) if isfile(join(filespath, f))])

    # constraints for which we have both vtrees and sdd
    existing_constraints = []
    for fuzzy in files:
        fuzzy_name = fuzzy.split(".")
        assert len(fuzzy_name) == 2
        assert fuzzy_name[1] == "fuzzy"

        constraint_name = fuzzy_name[0]
        existing_constraints.append(constraint_name)

    return existing_constraints



def sympy_object_to_tf_tree(sympy_obj):
    # For Paolo with <3
    pass