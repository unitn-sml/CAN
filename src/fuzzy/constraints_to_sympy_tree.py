import sys
import re as regex
import numpy as np
import ast
import sympy
from sympy.logic.boolalg import to_cnf, to_dnf, simplify_logic
from sympy.parsing.sympy_parser import parse_expr
import argparse
import pickle
import os

"""
have to be splitting sympy parsing in multile jobs because it can become very slow
after a certain length of the string to parse, also it can lead to a stack overflow
"""
from multiprocessing import Pool


class ConstraintsToSympyTree:
    """
    Class with only a public and static method to go from a file specifying constraints in
    propositional logic (sympy syntax) to DIMACS CNF files.
    The input file must specify the shape of the variables, so that the DIMCAS CNF file can be written
    by rewriting the constraints as if those variables were reshaped as a single, longer vector.
    This scripts takes into account the input shape to properly rename variables and rewrite the constraints
    specified in the input file as a DIMACS CNF file.
    Refer to the documentation of expression_to_cnf for more information.
    Usage:
        ConstraintsToCnf.expression_to_cnf(input file, output file)
    """

    @classmethod
    def expression_to_sympy_tree(cls, constraint_file, output_file, nprocesses=1, mode=None, simplify=False):
        """
        Given an input file specifying logical constraints with sympy syntax, one line
        after another, where each constraint is considered to be on an "and" relationship
        with all others (on all other lines).
        Lines starting with '#' are considered comments.
            This scripts takes into account the input shape to properly rename variables and rewrite the constraints
            specified in the input file as a DIMACS CNF file.

        :param constraint_file: Input file in sympy syntax specifying constraints, 1 per line, considered in an 'and'
        relationship between them.
            Lines starting with '#' are considered comments.
            The first non-comment line should be specifying the shape of the tensor in which the referred variables are.
            After that, all non-comment lines are assumed to be propositional logic constraints written in the sympy
            syntax. All these constraints (on different lines) are to be considered in an and relationship.
            Variables must be specified similarly as if we were indexing a tensor with multiple dimensions, of the same shape
            we specified earlier.
            This means that we can't refer to variables which have more "dimension" than the actual shape, and those
            indexing dimensions should respect the initially specified shape.

            example file:
            ------
            # comment: 4 variables, multinomial distribution, 3 possible states for each variable
            shape [4, 3]

            # this is a comment
            X.1.1 >>    X.2.2
            X3.1 >>    X2.2
            ------

            This file tells about variables set in a shape of [4,3], this means we will have to refer to each variable
            as X.<first dimension index>.<second dimension index>.
            Optionally, the dot ('.') for the first index can be skipped, this might help in clarity in case of shapes
            of 1 dimension, for example:
                shape [10]
                X1 | X2
                X.1 & X10
        :param output_file: Output DIMACS CNF file, variables here are considered/named/referred as if we had the
        same number of variables as specified in the shape of the input file, but reshaped as a single vector, think
        something like variables.reshape(-1).
            This scripts takes into account the input shape to properly rename variables and rewrite the constraints
            specified in the input file as a DIMACS CNF file.
        :param nprocesses: Number of processes to use during the parsing from string to simpy a sympy expression.
            An higher number of processes might be needed because sympy seems to have some problems
            after the length of a string to parse gets past an arbitrary threshold.
        """

        # parse data, get shape (list of ints) and constraints (list of strings)
        shape, constraints = ConstraintsToSympyTree._read_data(constraint_file)

        """
        Stride for each dimension: 
        get the shape as a list and append 1
        invert it and make it into a np array
        compute the cumulative product
        transform the result in a list, invert it, discard the first element, make it back into a numpy array
        we have achieved unreadability
        """
        shape = np.array(shape)
        stride = np.cumprod(np.append(shape, 1)[::-1])[::-1][1:]

        # note that this is the total number of variables in the mono dimensional array, nor in the constraints
        total_vars = np.prod(shape)

        """
        For the remaining items, that should be propositional algebra constraints,'and' them togheter.
        Note that this means that you can actually write stuff
        in the input file that is not part of propositional logic, but is part of what
        sympy allows (math in general), this will probably result in an error
        when this script will later convert everything to CNF, but it may happen that your
        expression is somewhat accepted and converted, and this will lead to unexpected behaviour.
        
        tldr: if you seek unexpected behaviour writing non logic stuff with sympy you will find it.
            We will 'and' each constraint, starting from a base constraint of 'true'
        """

        if nprocesses == 1:
            constraints_expression = ConstraintsToSympyTree._parse_constraints_monoprocess(constraints)
        else:
            constraints_expression = ConstraintsToSympyTree._parse_constraints_multiprocess(constraints, nprocesses)
        
        # check variables for correctness (shape wise and name/format)
        for var in constraints_expression.atoms():
            ConstraintsToSympyTree._assert_is_valid_variable(str(var), shape, stride, total_vars)

        if mode == "cnf":
            # now that we have our constraints, convert them to CNF
            print(f"converting to cnf with simplification: {simplify}")
            constraints_expression = to_cnf(constraints_expression, simplify=simplify)

        elif mode == "dnf":
            # now that we have our constraints, convert them to CNF
            print(f"converting to dnf with simplification: {simplify}")
            constraints_expression = to_dnf(constraints_expression, simplify=simplify)

        # write output files
        print("writing to sympy tree file")
        ConstraintsToSympyTree._to_sympy_tree(shape, stride, total_vars, constraints_expression, constraint_file, output_file, mode)

    @staticmethod
    def _parse_constraints_monoprocess(constraints):
        """
        Function that parses a list of string into a sympy expression.
        :param constraints: Constraints as a list of strings.
        :return: A simpy expression, the "and" of all the constraints.
        """
        sys.setrecursionlimit(1500000)

        parsed_constraints = []
        for constraint in constraints:
            print("Parsed constraint")
            assert "_" not in constraint, "Symbol '_' is reserved."

            # we need to substitute "." with _, since otherwise sympy will think this is a float number
            # evaluate is set to false to avoid evaluating to False clauses that we explicitly wanted to be
            # unsatisfiable
            parsed_constraints.append(parse_expr(constraint.replace(".", "_"), evaluate=False))

        # 'and' them together
        constraints_expression = sympy.And(*parsed_constraints, evaluate=False)

        return constraints_expression

    @staticmethod
    def _parse_constraints_multiprocess(constraints, nprocesses):
        """
        Function that parses a list of string into a sympy expression
        by using multiple processes.
        :param constraints: Constraints as a list of strings.
        :param nprocesses: Number of processes to use.
        :return: A simpy expression, the "and" of all the constraints.
        """

        # stackoverflow to split constraints to each process
        def chunks(a, n):
            k, m = divmod(len(a), n)
            return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

        # open pool of processes, each one will contribute to parsing part of the constraints
        with Pool(processes=nprocesses) as pool:
            multiple_results = [pool.apply_async(ConstraintsToSympyTree._parse_constraints_monoprocess, [chunk]) for chunk in
                                chunks(constraints, nprocesses)]

            constraints_list = [res.get() for res in multiple_results]
            constraints_expression = sympy.And(*constraints_list, evaluate=False)

        return constraints_expression

    @staticmethod
    def _read_data(constraint_file):
        """
        Read the data file, parsing the shape and the constraints.

        :param constraint_file: Input constraint file having the shape followed by constraints in sympy syntax.
        :return: shape as a list of ints, constraints as a list of strings
        """
        with open(constraint_file, 'r') as myfile:
            """
            read the file as a string, split by newlines
            clean out "" (empty lines) and comments)
            """
            data = list(filter(lambda x: x != "" and x[0] != "#", myfile.read().split("\n")))
            # clean out white space
            data = list(map(lambda x: regex.sub(r'\s+', ' ', x), data))

            # check that the first non-comment and non-empty line is the shape
            shapestring = data[0]
            assert shapestring[:6] == "shape "

            # make sure what comes after looks like a list of ints
            shapestring = shapestring[6:]
            assert regex.search(r"^\[(\s*[0-9]*\s*,)*\s*[0-9]*\s*\]$",
                                shapestring), "%s is not a valid shape" % shapestring[6:]

            # read it as a list of ints and obtain, and obtain the info we need about variables (total vars, stride etc)
            shape = ast.literal_eval(shapestring)
            constraints = data[1:]

            return shape, constraints

    @staticmethod
    def _assert_is_valid_variable(var, shape, stride, total_vars):
        """
        Assert this string is a valid variable in our syntax, and if its indexes respect
        the input shape.

        :param var: Variable as as string, in the form X.1.2.3 etc.
        :param shape: Shape in which we consider our variables to be.
        :param stride: Numpy array specifying the stride over each dimension.
        :param total_vars: Total number of variables (given by the shape).
        """
        assert bool(regex.search("^X((_[0-9]+)|[0-9]+)(_[0-9]+)*$",
                                 var)), "Variable '{}' does not conform to the supported syntax.".format(var)
        # get all indexes as ints
        items = list(map(lambda x: int(x), regex.findall("[0-9]+", var)))

        # specified indexes equal the number of dimensions
        assert len(items) == len(shape), \
            "Number of indexes in {} is not the same as the number of dimensions specified in shape '{}'".format(var, shape)
        # all indexes should be < the dimension they index
        for number, index_in_this_dim, dim in zip(list(range(len(items))), items, shape):
            assert index_in_this_dim < dim, "Index number {} '{}' in var '{}' should be < {}, given the shape {}".format(
                number, index_in_this_dim, var, dim, shape)

        """
        Make sure that once we shape our variables as [-1], the variable we would be indexing
        is not out of bounds.
        """
        assert ConstraintsToSympyTree._get_index(np.array(items), stride) < total_vars

    @staticmethod
    def _get_index(index, stride):
        """
        Given an index specified by a np array and a stride, again specified by
        a numpy array, compute the index that would result if we were to index a variable
        when all variables would be shaped as a single 1 dimensional vector.

        :param index: Numpy array specifying the indexed variable.
        :param stride: Numpy array specifying the stride over each dimension.
        :return: Index of the variable when variables are reshaped in a 1 dimensional vector.
        """
        return np.dot(stride, index)

    @staticmethod
    def _to_sympy_tree(shape, stride, total_vars, formula, input_file, output_folder, mode):
        """
        Writes the constraints expressed by the sympy expression
        as a pickle file, where all variables that were referred in the input_file
        as if being in a tensor of the specified shape are renamed as if they were
        in a 1 dimensional vector.

        :param shape: Shape in which we consider our variables to be.
        :param stride: Numpy array specifying the stride over each dimension.
        :param total_vars: Total number of variables (given by the shape).
        :param formula: Sympy boolean expression.
        :param input_file: Refer to expression_to_sympy_tree.
        :param output_file: Refer to expression_to_sympy_tree.
        """
        constraint_name = os.path.split(input_file)[-1].split(".")[0]

        # save file as pickle
        if mode == "dnf":
            pickle.dump(formula, open(os.path.join(output_folder, "{}_dnf.fuzzy".format(constraint_name)), "wb"))
        else:
            pickle.dump(formula, open(os.path.join(output_folder,
                                                   "{}.fuzzy".format(
                                                       constraint_name)), "wb"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True, 
                        help="Path to the input file")
    parser.add_argument("-o", "--output_folder", type=str, required=True, 
                        help="Path to the output folder")
    parser.add_argument("-p", "--nprocesses", type=int, required=False, default=1,
                        help="Number of processes to use")
    parser.add_argument("-m", "--mode", type=str, required=False, default=None, choices=["cnf", "dnf"],
                        help="One of CNF or DNF: apply the requested " \
                            "trasformation to the constraint formula")
    parser.add_argument("-s", "--simplify", action="store_true",
                        help="Simplify while transforming in CNF or DNF")
    args = parser.parse_args()

    assert os.path.isdir(args.output_folder), "Output folder {} does not exists".format(args.output_folder)
    assert os.path.isfile(args.input), "Input file {} does not exists".format(args.input)

    ConstraintsToSympyTree.expression_to_sympy_tree(args.input, args.output_folder, args.nprocesses, mode=args.mode, simplify=args.simplify)
