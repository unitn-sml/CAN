#from fuzzy.lyrics.lyrics.fuzzy import Lukasiewicz, LukasiewiczStrong, Goedel
import os
import pickle
import tensorflow as tf
import numpy as np
import sympy
import re as regex

from sympy.logic.boolalg import (
    And,
    Or,
    Not,
    Nand,
    Nor,
    Implies,
    Xor,
    Equivalent,
    BooleanAtom,
    BooleanFalse,
    BooleanTrue
)
from sympy import Symbol

"""
Given a formula saved as constraints_name.fuzzy
create a tf graph representing the formula expressed in fuzzy logic.
"""
class Parser:

    def __init__(self, experiment, sympy_formula, solver, is_file=True):
        self.experiment = experiment
        self.formula = pickle.load(open(sympy_formula, "rb")) if is_file else sympy_formula
        self.solver = solver

    def __str__(self):
        return "Parser on formula: " + str(self.formula)

    def generate_tf_tree(self, probabilities, use_dnf=False, simple_only=False):
        """
        use_dnf: @Pier mettici due right
        simply_only: use only Nor, And and Or operations
        """
        # create mapping between boolean variables and tensor strides
        self.var2nodes = Parser.variables_to_nodes(self.formula, probabilities)
        # build the tf tree recursively from the boolean formula
        res = self.recurse(self.formula, use_dnf, simple_only=simple_only)
        return res

    @staticmethod
    def variables_to_nodes(formula, tensor):
        """
        Create a dict mapping sympy variables to pieces of the input tensorflow tensor
        """
        # move batch size to the end
        #tensor = tf.transpose(tensor, perm=list(range(1, len(tensor.shape))) + [0])
        res = {}
        for var in formula.atoms():
            indexes = tf.constant(list(map(lambda x: int(x), regex.findall("[0-9]+", str(var)))))
            # indexes shape [k]
            # tensor shape [None, a_1, a_2, a_3, ..., a_k]
            indexes = tf.concat(
                [
                    tf.range(tf.shape(tensor)[0])[:, tf.newaxis], # [None, 1]
                    tf.tile(indexes[tf.newaxis, :], (tf.shape(tensor)[0], 1)) # [None, k]
                ],
                axis=-1
            )
            res[var] = tf.gather_nd(tensor, indexes)
        return res


    def recurse(self, node, use_dnf, simple_only):
        nargs = len(node.args)
                
        if isinstance(node, Symbol):
            return self.var2nodes[node]

        if isinstance(node, And):
            assert nargs > 1
            return self.solver.weak_conj(
                [self.recurse(x, use_dnf, simple_only) for x in node.args]
            )

        if isinstance(node, Or):
            assert nargs > 1
            return self.solver.strong_disj(
                [self.recurse(x, use_dnf, simple_only) for x in node.args]
            )

        if isinstance(node, Not):
            assert nargs == 1
            return self.solver.negation(
                self.recurse(node.args[0], use_dnf, simple_only)
            )

        if isinstance(node, Nand):
            assert nargs > 1
            return self.solver.negation(
                self.solver.weak_conj(
                    [self.recurse(x, use_dnf, simple_only) for x in node.args]
                )
            )

        if isinstance(node, Nor):
            assert nargs > 1
            return self.solver.negation(
                self.solver.strong_disj(
                    [self.recurse(x, use_dnf, simple_only) for x in node.args]
                )
            )

        if isinstance(node, Implies):
            assert nargs == 2
            if simple_only:
                return self.solver.strong_disj(
                    [
                        self.solver.negation(
                            self.recurse(node.args[0], use_dnf, simple_only)
                        ),
                        self.recurse(node.args[1], use_dnf, simple_only)
                    ]
                )
            else:
                return self.solver.implication(
                    self.recurse(node.args[0], use_dnf, simple_only),
                    self.recurse(node.args[1], use_dnf, simple_only)
                )
    
        if isinstance(node, Xor):
            assert nargs == 2
            if simple_only:
                # xor equivalent to : (A and ~B) or (~A and B)
                return self.solver.strong_disj([
                    self.solver.weak_conj([
                        self.recurse(node.args[0], use_dnf, simple_only),
                        self.solver.negation(self.recurse(node.args[1], use_dnf, simple_only))
                    ]),
                    self.solver.weak_conj([
                        self.solver.negation(self.recurse(node.args[0], use_dnf, simple_only)),
                        self.recurse(node.args[1], use_dnf, simple_only)
                    ])
                ])
            else:
                return self.solver.exclusive_disj(
                    [self.recurse(x, use_dnf, simple_only) for x in node.args]
                )

        if isinstance(node, Equivalent):
            assert nargs == 2
            if simple_only:
                # iff equivalent to : (~A and ~B) or (A and B)
                return self.solver.strong_disj([
                    self.solver.weak_conj([
                        self.solver.negation(self.recurse(node.args[0], use_dnf, simple_only)),
                        self.solver.negation(self.recurse(node.args[1], use_dnf, simple_only))
                    ]),
                    self.solver.weak_conj([
                        self.recurse(node.args[0], use_dnf, simple_only),
                        self.recurse(node.args[1], use_dnf, simple_only)
                    ])
                ])
            else:
                return self.solver.iff(
                    self.recurse(node.args[0], use_dnf, simple_only),
                    self.recurse(node.args[1], use_dnf, simple_only)
                )

