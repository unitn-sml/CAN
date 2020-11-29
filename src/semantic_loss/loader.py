import os
from os.path import isfile, join
from itertools import islice
import random
import numpy as np
import scipy.misc
import scipy.ndimage

"""
Loaders for specific configs, dataset and image plotting for different stuff.
Check the experiment class and its README to understand why those are needed and how
they are used.
"""


def _get_images(directory):
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            abspath = os.path.abspath(join(dirpath, f))
            if isfile(abspath):
                img = scipy.ndimage.imread(abspath, flatten=True)
                img[img <= 127] = 0
                img[img > 127] = 1
                img = img.astype(np.int32)
                assert np.max(img) == 1 or np.max(img) == 0
                assert np.min(img) == 0
                values = len(set(list(np.reshape(img, -1))))
                assert values == 2 or values == 1
                yield (abspath.split("/")[-1], img)


def _get_formulas(directory):
    import csv
    with open(directory, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        formulas = []
        for formula in csv_reader:
            formulas.append(np.asarray(formula).astype(int))
    return formulas


def _reshape_data(samples, shape):
    return np.reshape(np.stack(samples), newshape=(-1, shape[0], shape[1], 1))


def _reshape_formulas(samples, shape):
    return np.reshape(np.stack(samples), newshape=(-1, shape[0]))


def get_config_mnist_still_life(config_dict):
    res = dict()
    res["DATASET_NAME"] = config_dict["DATASET_TYPE"]

    res["SEMANTIC_LOSS_CONSTRAINTS_DROPOUT_RATE"] = config_dict.get("SEMANTIC_LOSS_CONSTRAINTS_DROPOUT_RATE", 0.00)
    drvalue = res["SEMANTIC_LOSS_CONSTRAINTS_DROPOUT_RATE"]
    assert isinstance(drvalue, float)
    # assert 0 <= drvalue <= 1

    res["SEMANTIC_LOSS_CONSTRAINTS_DROPOUT_DECREMENTAL"] = config_dict.get(
        "SEMANTIC_LOSS_CONSTRAINTS_DROPOUT_DECREMENTAL", False)
    assert isinstance(res["SEMANTIC_LOSS_CONSTRAINTS_DROPOUT_DECREMENTAL"], bool)

    wsize = config_dict.get("STILL_LIFE_WINDOW_SIZE", None)
    assert wsize is None or (isinstance(wsize, int) and wsize > 0)
    res["STILL_LIFE_WINDOW_SIZE"] = wsize

    res["SL_EQUALIZE"] = config_dict.get("SL_EQUALIZE", False)
    assert isinstance(res["SL_EQUALIZE"], bool)

    return res


def get_config_random_formulas(config_dict):
    res = dict()
    res["DATASET_NAME"] = config_dict["DATASET_TYPE"]
    res["FORMULA_FILE"] = config_dict["FORMULA_FILE"]
    res["FOLDER_DATASET"] = config_dict["FOLDER_DATASET"]
    return res


def get_dataset_formulas(dataset_folder, formula_file, splits, shape):
    def get_dataset_wrapped():
        dataset_name = "dataset_%s"

        # should be refactored in a cleaner way
        dataset_name = dataset_name % formula_file

        path = f"in/datasets/random_formulas/{dataset_folder}/{dataset_name}.csv"
        dataset = list(_get_formulas(path))
        random.seed(1337)
        random.shuffle(dataset)
        dataset = iter(dataset)

        training, test, validation = (list(islice(dataset, 0, i))
                                      for i in splits)

        return _reshape_formulas(training, shape) if len(training) > 0 else [], \
               _reshape_formulas(test, shape) if len(test) > 0 else [], \
               _reshape_formulas(validation, shape) if len(validation) > 0 else []

    tr, t, d = get_dataset_wrapped()

    return get_dataset_wrapped


def get_dataset_mnist_parity_check(name, splits, shape):
    def get_dataset_wrapped():
        dataset_name = "MNIST_PARITY_CHECK_%s"

        # should be refactored in a cleaner way
        if shape == [26, 26, 1]:
            dataset_name = dataset_name % (name.split("_")[-2])
            dataset_name = dataset_name + "_26x26"
        else:
            dataset_name = dataset_name % (name.split("_")[-1])

        path = "in/datasets/" + dataset_name
        dataset = list(_get_images(path))
        dataset = sorted(dataset, key=lambda x: x[0])
        random.seed(1337)
        random.shuffle(dataset)
        dataset = [data[1] for data in dataset]
        dataset = iter(dataset)

        training, test, validation = (list(islice(dataset, 0, i))
                                      for i in splits)

        return _reshape_data(training, shape) if len(training) > 0 else [], \
               _reshape_data(test, shape) if len(test) > 0 else [], \
               _reshape_data(validation, shape) if len(validation) > 0 else []

    tr, t, d = get_dataset_wrapped()

    return get_dataset_wrapped



def get_dataset_mnist_still_life(name, splits, shape):
    def get_dataset_wrapped():
        dataset_name = "mnist_still_dataset_%s"

        # should be refactored in a cleaner way
        if shape == [26, 26, 1]:
            dataset_name = dataset_name % (name.split("_")[-2])
            dataset_name = dataset_name + "_26x26"
        else:
            dataset_name = dataset_name % (name.split("_")[-1])

        path = "in/datasets/mnist_still_life/" + dataset_name
        dataset = list(_get_images(path))
        dataset = sorted(dataset, key=lambda x: x[0])
        random.seed(1337)
        random.shuffle(dataset)
        dataset = [data[1] for data in dataset]
        dataset = iter(dataset)

        training, test, validation = (list(islice(dataset, 0, i))
                                      for i in splits)

        return _reshape_data(training, shape) if len(training) > 0 else [], \
               _reshape_data(test, shape) if len(test) > 0 else [], \
               _reshape_data(validation, shape) if len(validation) > 0 else []

    tr, t, d = get_dataset_wrapped()

    return get_dataset_wrapped

def get_dataset_mnist(name, splits, shape):
    def get_dataset_wrapped():
        dataset_name = "mnist_dataset_%s"

        # should be refactored in a cleaner way
        dataset_name = dataset_name % (name.split("_")[-1])

        path = "in/datasets/mnist_still_life/" + dataset_name
        dataset = list(_get_images(path))
        dataset = sorted(dataset, key=lambda x: x[0])
        random.seed(1337)
        random.shuffle(dataset)
        dataset = [data[1] for data in dataset]
        dataset = iter(dataset)

        training, test, validation = (list(islice(dataset, 0, i))
                                      for i in splits)

        return _reshape_data(training, shape) if len(training) > 0 else [], \
               _reshape_data(test, shape) if len(test) > 0 else [], \
               _reshape_data(validation, shape) if len(validation) > 0 else []

    tr, t, d = get_dataset_wrapped()

    return get_dataset_wrapped


def plot_images_mnist_still_life(output_folder, shape):
    def plot_images_wrapped(samples, epoch):
        # samples shape: (eval_samples, width, height, channels)
        image_path = output_folder + "{}_{}.png"
        for j, image in enumerate(samples):
            image = image.reshape((shape[0], shape[1]))
            scipy.misc.imsave(image_path.format(
                str(epoch).zfill(6), str(j).zfill(2)), image)

    return plot_images_wrapped

def plot_random_formulas(output_folder, shape):
    def mkdir(path):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

    def plot_images_wrapped(samples, epoch):
        output_folder_samples = os.path.join(output_folder, f"samples_epoch_{epoch}")
        output_file = os.path.join(output_folder_samples, "samples.csv")
        mkdir(os.path.join(output_folder, f"samples_epoch_{epoch}"))
        np.savetxt(output_file, np.asarray(samples > 0.5), fmt='%s', delimiter=",")

    return plot_images_wrapped
