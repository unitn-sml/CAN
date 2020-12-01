"""
Module containing utility stuff for the semantic_loss directory.
"""
import os
from os import listdir
from os.path import isfile, join


def _get_constraints_names():
    """
    Get the names of all constraints for which there is both a vtree and a sdd file in
    in/semantic_loss_constraints/constraints_as_vtree  (as_sdd).
    Expect the cwd to be the /src directory of the repository.
    :return: List of strings indicating names of constraints that can be instantiated.
    """
    cwd = os.getcwd()
    assert cwd[-4:] == "/src", "Expected to be in the src directory of the repository, the cwd is instead %s" % cwd

    vtreepath = "in/semantic_loss_constraints/constraints_as_vtree"
    vtrees = sorted([f for f in listdir(vtreepath) if isfile(join(vtreepath, f))])

    sddpath = "in/semantic_loss_constraints/constraints_as_sdd"
    sdds = set([f for f in listdir(sddpath) if isfile(join(sddpath, f))])

    if len(sdds) != len(vtrees):
        print("Warning, some constraints have an existing vtree/sdd file, but not both (either vtree or sdd file is "
              "missing)")

    # constraints for which we have both vtrees and sdd
    existing_constraints = []
    for vtree in vtrees:
        sdd_name = vtree.split(".")
        assert len(sdd_name) == 2
        assert sdd_name[1] == "vtree"

        constraint_name = sdd_name[0]
        sdd_name = sdd_name[0] + ".sdd"
        if sdd_name in sdds:
            existing_constraints.append(constraint_name)
    return existing_constraints
