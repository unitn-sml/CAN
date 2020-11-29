import sys
sys.path.append("..") # Adds higher directory to python modules path.

import argparse
import os
import tensorflow as tf
from fuzzy.lyrics import fuzzy as fuzzy_circuits
from utils import utils_common
import numpy as np
import csv

base_path = os.path.join("..", "in", "semantic_loss_constraints")

def import_sympy(name, class_name):
    from fuzzy.parser import Parser

    sympy_file = os.path.join(base_path, "constraints_as_sympy_tree", f"{name}.fuzzy")
    assert os.path.isfile(sympy_file), f"{sympy_file} is not a file."

    # load vtree and sdd files and construct the PSDD
    res = Parser(None, sympy_file, class_name)

    return res


def import_psdd(name):
    from thirdparties.py3psdd import Vtree, SddManager, PSddManager, io

    vtree = os.path.join(base_path, "constraints_as_vtree", f"{name}.vtree")
    sdd = os.path.join(base_path, "constraints_as_sdd", f"{name}.sdd")

    assert os.path.isfile(vtree), f"{vtree} is not a file."
    assert os.path.isfile(sdd), f"{sdd} is not a file."

    # load vtree and sdd files and construct the PSDD
    vtree = Vtree.read(vtree)
    manager = SddManager(vtree)
    alpha = io.sdd_read(sdd, manager)
    pmanager = PSddManager(vtree)
    psdd = pmanager.copy_and_normalize_sdd(alpha, vtree)

    return psdd


if __name__ == "__main__":

    argparser = argparse.ArgumentParser(description=
        """
        This is a test of the command line argument parser in Python.
        """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    argparser.add_argument("-c", "--circuit_name", type=str, required=True, help="Input circuit name generated with fuzzy/constraints_to_sympy_tree.py (fl) or semantic_loss/constraints_to_cnf.py and pysdd -w -d")
    argparser.add_argument("-t", "--test_file", type=str, required=True, help="Test file with a probability vector on each line")
    
    argparser.add_argument("-m", "--mode", type=str, required=True, choices=["sl", "fl"])
    argparser.add_argument("-f", "--fuzzy_logic_class", type=str, required=False, default="Lukasiewicz", help="FL class to be used")

    argparser = argparser.parse_args()

    circuit_name = argparser.circuit_name
    test_file = argparser.test_file
    mode = argparser.mode
    fuzzy_logic_class = argparser.fuzzy_logic_class

    print("Creating placeholder")
    # aggregate has shape: [batch_size, 2, 2]
    aggregate = tf.placeholder(shape=[None, 2, 2], dtype=tf.float32)
    wmc = None

    # apply SL or FL tree
    if mode == "fl":
        print("Importing FL parser")
        fuzzy_classes = utils_common.get_module_classes(fuzzy_circuits.__name__)
        assert fuzzy_logic_class in fuzzy_classes

        fuzzy_class = fuzzy_classes[fuzzy_logic_class]

        # Parsing input formula
        parser = import_sympy(circuit_name, fuzzy_class)

        print(f"Generating TF tree from formula in {mode}")
        # generate tf tree that encodes fuzzy logic loss
        wmc = parser.generate_tf_tree(aggregate)

    else:
        print("Importing SL parser")
        parser = import_psdd(circuit_name)

        aggregate2 = tf.reshape(aggregate, [-1, 4])
        wmc = parser.generate_tf_ac_v2(aggregate2)


    # instantiate session
    sess = tf.Session()

    # readtsv file
    data = []

    with open(test_file) as ff:

        for row in ff.readlines():
            row = row.strip()
            # avoid comments
            if row.startswith("#"):
                continue
            row = row.split("\t")
            row = [x for x in row if x]
            if len(row) > 0:
                assert len(row) == 4, f"Line '{row}' does not have length 4"
                data.append([float(x) for x in row])

    data = np.array(data)
    data = np.reshape(data, (-1, 2, 2))

    res = sess.run(wmc, feed_dict={aggregate: data})
    
    print("Results:")
    for r in res:
        print(r)
    