"""
======
PyPSDD
======

    PyPSDD is a Python package for inference and learning with PSDDs.

    https://github.com/art-ai/pypsdd
"""
__author__ = ("Arthur Choi <aychoi@cs.ucla.edu>")
__license__ = "Apache License, Version 2.0"
__date__    = "July 4, 2018"
__version__ = "0.1"
__bibtex__ = ("@inproceedings{KisaVCD14,\n"
              "author = {Doga Kisa and Guy {Van den Broeck} and "
              "Arthur Choi and Adnan Darwiche},\n"
              "title = {Probabilistic Sentential Decision Diagrams},\n"
              "booktitle = {Proceedings of the 14th International "
              "Conference on Principles of Knowledge Representation "
              "and Reasoning (KR)},\n"
              "year = {2014}\n"
              "}")

from py3psdd.vtree import Vtree
from py3psdd.manager import SddManager,PSddManager
from py3psdd.sdd import SddNode
from py3psdd.psdd import PSddNode
from py3psdd.prior import Prior,DirichletPrior,UniformSmoothing
from py3psdd.data import DataSet,Inst,InstMap
from py3psdd.timer import Timer
from py3psdd import io

__all__ = ["Vtree","SddManager","PSddManager", \
           "SddNode","PSddNode", \
           "Prior","DirichletPrior","UniformSmoothing", \
           "DataSet","Inst","InstMap", \
           "Timer"]
