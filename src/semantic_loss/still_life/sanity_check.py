"""
Simple and naive script to check that all
pictures are correct still lifes
"""
import os
from os.path import isfile, join
import numpy as np
import scipy.misc
import scipy.ndimage
from scipy.ndimage import convolve


def get_images(directory):
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


datasets = ["mnist_still_dataset_10000", "mnist_still_dataset_20000"]
datasets = [list(get_images(dataset)) for dataset in datasets]

# use this to sum the neighbours
kernel = np.array(np.array([[1, 1, 1],
                            [1, 0, 1],
                            [1, 1, 1]]))
for dataset in datasets:
    for datapoint in dataset:
        name, data = datapoint
        sum_neighbours = convolve(data, kernel, mode='constant')
        assert len(data.shape) == 2

        # sum of neighbours of elements which are zeros
        zeros_sum = sum_neighbours[data == 0]

        # sum of neighbours of elements which are ones
        ones_sum = sum_neighbours[data == 1]

        # check that shapes are correct
        assert (data.shape[0] * data.shape[1]) == zeros_sum.shape[0] + ones_sum.shape[
            0], "Shape problem with datapoint % s" % name

        # assert no sum of neighbours of zeros is 3
        assert np.all(zeros_sum != 3), "Zero neighbours sums problem with datapoint %s" % name

        # assert all sums of neighbours of ones are either 2 or 3
        assert np.all((ones_sum == 3) | (ones_sum == 2)), "One neighbours sums problem with datapoint %s" % name
