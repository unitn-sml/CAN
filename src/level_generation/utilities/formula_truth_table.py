from itertools import product
from sympy import Lambda
import sys
import pickle

assert len(sys.argv) == 2

formula = pickle.load(open(sys.argv[1], "rb"))
atoms = tuple(formula.atoms())

ff = Lambda(atoms, formula)

for p in product((True, False), repeat=len(atoms)):
    doc = {k: v for k, v in zip(atoms, p)}
    res = ff(*p)
    print(f"Assignment {p} resulted in {res}")