"""
Module to generate random constraints and to
extract a certain number of models from it.

Output is returned in a directory, with the constraints
written as 1 per line constraint (as needed by the constraints_to_cnf module),
a file containing data, 1 per line.
"""
import argparse
import itertools
import os
import random

import sympy
from sympy import symbols


def generate_clause(vars, operator):
    """
    Generate a clause by applying the operator to the vars.
    :param vars: list of sympy symbols.
    :param operator: Sympy logical operator.
    :return: Sympy operator applied to input vars.
    """
    return operator(*vars, evaluate=False)


def random_apply_not(var):
    """
    Randomly apply the sympy.Not operator to a sympy var.
    :param var:
    :return:
    """
    if random.randint(0, 1):
        return var
    else:
        return sympy.Not(var)


def grouper(n, iterable):
    """
    From stackoverflow.
    Groups stuff from an iterator in chunks and returns it in a list.

    :param n:
    :param iterable:
    :return:
    """

    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def generate_constraint(nvars, nclauses, maxvars, verbose, multinomial, simplify):
    """
    Generate a cnf constraint given the parameters, return the variables used, the generated constraints
    as as sympy expression and what to output to file.

    :param nvars:
    :param nclauses:
    :param maxvars:
    :param multinomial:
    :param simplify:
    :return:
    """
    # get needed symbols
    print("Getting symbols")

    nvars *= multinomial
    vars = ["X%s" % i for i in range(nvars)]
    vars = " ".join(vars)
    vars = symbols(vars)

    # result that we will write to file
    result_as_string = ["shape [%s]" % nvars]

    # generate Or clauses
    clauses = []

    # first let's make sure that all variables appear in the clause (this is handy to later generate model that
    # in which we are sure all variables appear)
    for var in vars:
        # and with current constraint
        clauses.append(sympy.Or(var, sympy.Not(var), evaluate=False))

    # if needed, express that multinomial states are exclusive of each other
    if multinomial > 1:
        print("Adding exclusivty constraints among different states of multinomial variables")
        # for each possible states of a var
        for multinomial_states in grouper(multinomial, vars):
            # make a set out of this group
            states_set = set(multinomial_states)

            # for each state say that state -> Nor( other states)
            for var in multinomial_states:
                states_set.remove(var)
                clauses.append(sympy.Implies(var, sympy.Nor(*states_set)))
                states_set.add(var)

    print("Generating clauses")
    for _ in range(nclauses):
        indexes = [random.randrange(0, len(vars)) for _ in range(random.randint(1, maxvars))]
        in_clause = [random_apply_not(vars[index]) for index in indexes]
        clause = generate_clause(in_clause, sympy.Or)
        clauses.append(clause)

        # append this clause to file string
        result_as_string.append(str(clause))

        if verbose:
            print("Wrote %s/%s constraints." % (len(result_as_string) - 1, nclauses))

    print("Anding clauses")
    constraints = sympy.And(*clauses, evaluate=False)

    if simplify:
        print("Simplifying")
        constraints = sympy.to_cnf(constraints, True)
    elif multinomial > 1:
        # need to set it as a cnf because of the exclusivity constraints we added
        constraints = sympy.to_cnf(constraints, False)

    result_as_string = "\n".join(result_as_string)
    return vars, constraints, result_as_string


def output_result(nvars, nclauses, maxvars, n_models, seed, verbose, multinomial, simplify):
    """
    Generate constraints on nvars, with nclauses, each clause
    will have a minimum of 1 var and a max of maxvars, after that sample n_models
    and write constraints and models in a directory called output_named, in files called
    output_name.sympy and output_name.csv respectively.

    :param nvars:
    :param nclauses:
    :param maxvars:
    :param n_models:
    :param multinomial
    :param simplify
    :return:
    """

    output_name = "SAT_seed%s_variables%s_clauses%s_maxperclause%s_nmodels%s_multinomial%s_simplify%s" % (
        seed, nvars, nclauses, maxvars, n_models, multinomial, simplify)
    if not os.path.exists(output_name):
        os.makedirs(output_name)
    else:
        print("Warning, a directory with that name already exists, files will be overwritten")

    vars, constraint, result_as_string = generate_constraint(nvars, nclauses, maxvars, verbose, multinomial, simplify)

    print("Writing constraints to file")
    with open("%s/%s.sympy" % (output_name, output_name), "w") as output_file:
        output_file.write(result_as_string)

    print("Checking if constraints are satisfiable")
    # sample models, write them to file
    models = sympy.satisfiable(constraint, algorithm='dpll2', all_models=True)

    print("Sampling models")
    # write models as strings and have a list of them
    models_as_strings = []
    for model in itertools.islice(models, n_models):
        # need to check if there was a satisfiable model (otherwise the function returns a single "False", ~ clunky)
        if model:
            model_string = []
            for var in vars:
                model_string.append(str(int(model[var])))
            model_string = ",".join(model_string)
            models_as_strings.append(model_string)
        if verbose:
            print("Sampled %s models" % len(models_as_strings))

    print("Writing models to file")
    models_as_strings = "\n".join(models_as_strings)
    with open("%s/%s.csv" % (output_name, output_name), "w") as output_file:
        output_file.write(models_as_strings)

    # write config that generated this result
    with open("%s/%s.config" % (output_name, output_name), "w") as output_file:
        config = "Experiment generated by:\n %s" % (str(args)
                                                    .replace("Namespace", "")
                                                    .replace(",", "\n")
                                                    .replace("(", "")
                                                    .replace(")", ""))
        output_file.write(config)


if __name__ == "__main__":
    """
    Generate a directory containing 3 files:
    - <name>.config, the args that generated this directory
    - <name>.sympy constraints written as a CNF, 1 per line, sympy syntax, that can be translated to dimacs using the
        constraints_to_cnf.py script in this module
    - <name>.csv sampled models that satisfy the constraints, in a csv format, 1 model per line
    
    Arguments:
     -c = OR clauses in the cnf constraints, (might actually result in less than this if randomly the same clause is 
        generated.
     -v = Number of variables we are working on/that we are considering. Clauses will be randomly created by randomly
        Oring random variables (after possibly and randomly using the not operator on them).
     -m = Max number of literals per clause, going randomly from 1 to m.
     --nmodels= Number of models to sample, might result in less if there are less than --nmodels models.
     -s = Seed to use for random, 1337 by default.
     --verbose = Be verbose, might be useful in case you want to see the progress on generating clauses, sampling
        models, etc.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--clauses", type=int, required=True, help="Number of clauses")
    parser.add_argument("-v", "--variables", type=int, required=True, help="Number of variables")
    parser.add_argument("-m", "--max_per_clause", type=int, required=True, help="Max number of variables per clauses")
    parser.add_argument("--nmodels", type=int, required=True, help="Max number of models to sample")
    parser.add_argument("-s", "--seed", type=int, required=False, default=1337, help="Seed for random")
    parser.add_argument("--verbose", help="Be verbose.", action="store_true", required=False)
    parser.add_argument("--simplify", help="Try to simplify the cnf (will take way longer).", action="store_true",
                        required=False)

    multinomialmsg = "If multinomial is provided, (>= 2), variables are now considered to represent different states" \
                     "related to variables in a multinomial distribution.\n Considering an array of length --variables, " \
                     "each chunk of length --multinomial will contain variables considered to be related, different" \
                     "states of a multinomial distribution. \n This implies that on top of the --clauses randomly generated" \
                     "constraints, there will also be constraints specifying the fact that only 1 of those variables can" \
                     "be true at each moment.\n" \
                     "Note that, for example, providing --variables=10 and --multinomial=2 would result in a total" \
                     "of 10*2 variables, because in this case you are basically saying that you want 10 multinomial" \
                     "variables with 2 possible states.\n."
    parser.add_argument("--multinomial", type=int, required=False, default=1, help=multinomialmsg)
    args = parser.parse_args()

    random.seed(args.seed)

    assert args.clauses > 0
    assert args.variables > 0
    assert args.max_per_clause > 1
    assert args.nmodels >= 0
    assert args.multinomial >= 1

    output_result(args.variables, args.clauses, args.max_per_clause, args.nmodels, args.seed, args.verbose,
                  args.multinomial, args.simplify)
