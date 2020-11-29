import dill
import logging
import numpy as np
import os
import pickle
import random
import sys
import collections
import tensorflow as tf
from shutil import make_archive
from os import makedirs
from logging.handlers import MemoryHandler

from inspect import getmembers, isfunction, isclass
from numba.targets.registry import CPUDispatcher as numba_compiled_fn


def compress_folder(folder):
    make_archive(folder, "zip", folder)


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


def init_environment(experiment):
    set_seeds(experiment["ANN_SEED"])  # make experiments reproducible

    for f in [experiment["OUTPUT_FOLDER"], experiment["CHECKPOINT_FOLDER"],
              experiment["PICKLE_FOLDER"], experiment["TENSORBOARD_ROOT"]]:
        makedirs(f, exist_ok=True)


def first_arg_null_safe(f):
    # call f only if its first argument is not None.
    return lambda *fargs, **fkwargs: f(*fargs, **fkwargs) if fargs[0] else None


def set_or_get_logger(name, log_file, console_level=None, file_level=None, msg_fmt=None, date_fmt=None, capacity=50):
    """
    Returns a logger with the given name. If the logger does not already exists then it is created and console
    and file logging are setup according to the logging level arguments and the file name.
    If the logger exists already, if it is already a console and/or file logger, then the logger is returned as is,
    otherwise console and/or file logging are addend accordingly to the arguments.
    :param name: Name of the logger.
    :param log_file: File on where to write logs.
    :param console_level: Console level of logging. Default None, if None no console logging takes place.
    :param file_level: File level of logging. Default None, if None no file logging takes place.
    :param msg_fmt: If None, [asctime, filename, funcname, linenumber, levelname) is used.
    :param date_fmt: If None [d/m/Y H:M:S] is used.
    :param capacity: Capacity of the logger, messages are sent/flushed every time the capacity is reached.
    :return: A logger with the given name.
    """

    if msg_fmt is None:
        msg_fmt = '[%(asctime)s] {%(filename)s:%(funcName)s:%(lineno)d} %(levelname)s: %(message)s'
    if date_fmt is None:
        date_fmt = '%d/%m/%Y %H:%M:%S'

    formatter = logging.Formatter(msg_fmt, date_fmt)

    # first get logger, then updated its handlers
    logger = logging.getLogger(name)
    handlers = logger.handlers

    # if the logger was already setup just return it
    if len(handlers):
        print("Logger %s was already setup, not adding console and file handlers again." % name)
        return logger

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # file logging
    if file_level is not None:
        filehandler = logging.FileHandler(log_file)
        filehandler.setLevel(file_level)
        filehandler.setFormatter(formatter)
        filehandler = MemoryHandler(capacity=capacity, target=filehandler)
        logger.addHandler(filehandler)

    # console logging
    if console_level is not None:
        consolehandler = logging.StreamHandler()
        consolehandler.setLevel(console_level)
        consolehandler.setFormatter(logging.Formatter(msg_fmt, date_fmt))
        consolehandler = MemoryHandler(capacity=capacity, target=consolehandler)
        logger.addHandler(consolehandler)

    return logger


# Get all functions and classes that does not start with an underscore from a given python file
def get_module_functions(module_name):
    def is_numba_compiled_fn(o):
        return type(o) == numba_compiled_fn

    def is_module_function(o):
        return (is_numba_compiled_fn(o) or (isfunction(o))) and o.__module__ == module_name and o.__name__[0] != "_"

    return dict(getmembers(sys.modules[module_name], is_module_function))


def get_module_classes(module_name):
    """
    Import all public classes from a module.
    :param module_name: Name of the module to import public classes from.
    :return: Dict mapping class names to the class.
    """

    def is_module_class(o):
        return isclass(o) and o.__module__ == module_name and o.__name__[0] != "_"

    return dict(getmembers(sys.modules[module_name], is_module_class))


# Save and restore numpy and random state, to restore number generation from a given point

def load_random_states(experiment, epoch_folder):
    random.setstate(unpickle_binary_file(epoch_folder + experiment["PY_RANDOM_STATE"]))
    np.random.set_state(unpickle_binary_file(epoch_folder + experiment["NP_RANDOM_STATE"]))


def save_random_states(experiment, epoch_folder):
    pickle_binary_file(epoch_folder + experiment["PY_RANDOM_STATE"], random.getstate())
    pickle_binary_file(epoch_folder + experiment["NP_RANDOM_STATE"], np.random.get_state())


# Dill/Undill binary file
def dill_binary_file(file_name, item):
    with open(file_name, mode="wb") as output_f:
        dill.dump(item, output_f)


def undill_binary_file(file_name):
    with open(file_name, mode="rb") as input_f:
        return dill.load(input_f)


# Pickle save and load binary file
def pickle_binary_file(file_name, item):
    with open(file_name, mode="wb") as output_f:
        pickle.dump(item, output_f)


def unpickle_binary_file(file_name):
    with open(file_name, mode="rb") as input_f:
        return pickle.load(input_f)


# Load/Save to/from file
def save_as_text_file(file_name, content):
    with open(file_name, mode="w") as output_f:
        output_f.write(str(content))


def load_from_text_file(file_name):
    with open(file_name, mode="r") as input_f:
        return input_f.readline()


def remove_folder(folder):
    # maybe use this ??
    '''
    import shutil
    shutil.rmtree(examples_folder)
    '''
    # recursively remove folders
    for root, dirs, files in os.walk(folder, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))


def min_max(items):
    return min(items), max(items)


def update_dict_of_lists(d_to_update, **d):
    """Updates a dict of list with kwargs.

    Args:
        d_to_update (dict): dictionary of lists.
        **d: keyword arguments to append.

    """
    for k, v in d.items():
        if k in d_to_update.keys():
            d_to_update[k].append(v)
        else:
            d_to_update[k] = [v]


def safe_merge(dict1, dict2):
    """
    Merge two dicts, asserting that they do not share any key.

    :param dict1:
    :param dict2:
    :return: Merged dict resulting by the merge of the two input dicts.
    """
    for key in dict1:
        assert key not in dict2, "%s is part of both dictionaries, cannot merge the two dictionaries." % key

    for key in dict2:
        assert key not in dict1, "%s is part of both dictionaries, cannot merge the two dictionaries." % key

    # merge them if it's safe
    return {**dict1, **dict2}


class __ReadOnlyWrapper(collections.Mapping):
    """
    To make dicts read only (stackoverflow).

    """

    def __init__(self, data):
        self._data = data

    def __getitem__(self, key):
        return self._data[key]

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def __str__(self):
        return str(self._data)


def make_read_only(dict):
    """
    Make a dictionary into a new read only dictionary.
    :param dict:
    :return:
    """
    return __ReadOnlyWrapper(dict)
