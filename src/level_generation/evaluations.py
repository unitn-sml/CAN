"""
Take a folder of samples and compute Validity, Novelty and Uniqueness metrics.
"""
import os
import argparse
import numpy as np
import json
from functools import reduce


class Constraint:
    """
    base class for evaluating constraints
    requires only that a _get_wmc method is implemented bu subclasses
    """
    def _get_wmc(self, samples):
        """
        This method should be overridden
        :return: a numpy array of 0 and 1 with shape [batch_size, n_constraints_per_samples]
        """
        raise NotImplementedError("Should be implemented by subclasses")


class ReachabilityConstraint(Constraint):
    """
    Reachability defined on tiles bottom sx and dx
    """
    def __init__(self, mario_settings_file):
        # mario settings
        mario_settings = json.load(open(mario_settings_file))
        self.jumps = mario_settings['jumps']
        self.solid = mario_settings['solid_indexes']

    def _get_wmc(self, samples):
        assert len(
            samples.shape) == 4, "expecting samples to have shape [batch_size, height, width, channels]"
        # extract solid tiles
        samples = np.sum(np.take(samples, self.solid, axis=-1), axis=-1)
        samples = samples.astype(bool)
        # compute per sample constraint satisfaction
        return self._batch_reachability(samples)

    def _batch_reachability(self, x):
        from .reachability import find_reachable
        # x.shape = (None, height, width)
        reachabilities = []
        for sample in x:
            sample = find_reachable(sample, self.jumps)
            # consider 3x2 rectangles bottom left and right
            sample = self._reachability(sample)
            reachabilities.append(sample)
        return np.stack(reachabilities, axis=0)

    def _reachability(self, sample):
        return (sample[-3, 0] * (1 - sample[-2, 0]) + sample[-2, 0] * (1 - sample[-1, 0])), \
            (sample[-3, -1] * (1 - sample[-2, -1]) +
             sample[-2, -1] * (1 - sample[-1, -1]))


class AstarConstraint(Constraint):
    """
    Reachability defined on A* agent, requires a txt files with the results
    """
    def __init__(self, results_file):
        # mario settings
        self.results = []
        for line in open(results_file, 'r').readlines():
            line = line.strip()
            if line:
                self.results.append(-int(float(line)))

    def _get_wmc(self, samples):
        return np.reshape(np.array(self.results), [-1, 1])


class ReachabilityMetrics:
    """
    Base class for all VNU (validity, novelty, uniqueness) statistics computed on already discretized samples.
    Validity is computed using the related semantic loss to decide which samples are valid and which ones are not,
    and is equal to the ratio of valid samples to total samples, valid/total.
    Novelty is computed by the ratio of valid samples not in training_data to the total valid samples,
    new valid/total valid.
    Uniqueness is computed by the ratio of unique valid samples to the total valid samples, unique valid/total valid.
    Given the related semantic loss, an instance of this class will add the following statistics to tensorboard:
    - total valid items, and validity
    - total novel items, and novelty
    - total unique items, and uniqueness

    Level must be saved in numpy format with shape [height, width, n_channels] in onehot encoding!
    """
    def __init__(self, constraint):
        self.constraint = constraint

    def _load_from_folder(self, folder):
        res = []
        for filename in os.listdir(folder):
            filename = os.fsdecode(filename)
            if filename.endswith(".npy"):
                sample = np.load(os.path.join(folder, filename))
                res.append(sample)            
        res = np.stack(res, axis=0)
        # returning an array with shape [number_of_samples, height, width, n_channels]
        return res

    def _evaluator(self, samples_folder, training_data_folder):
        """
        Load the samples from the two folder and compute the VNU statistics
        """
        samples = self._load_from_folder(samples_folder)
        training_data = self._load_from_folder(training_data_folder)
        # return statistics on Validity, Novelty and Uniqueness
        return self._vnu_stats(training_data, samples)

    def _vnu_stats(self, training_samples, eval_samples):
        """
        :param training_samples:
        :param eval_samples:
        :return:
        """

        eval_samples = eval_samples.astype(float)
        training_samples = training_samples.astype(float)

        print("Computing validity")
        wmc = self.constraint._get_wmc(eval_samples)
        # product between constraint of the same level to find perfect items
        wmc_per_sample = np.prod(wmc, axis=-1)

        # reshape training data and samples to [bs, -1]
        training_samples = np.reshape(training_samples, [training_samples.shape[0], -1])
        eval_samples = np.reshape(eval_samples, [eval_samples.shape[0], -1])

        # keep only unique training data
        training_samples = np.unique(training_samples, axis=0)

        # indicates if a sample is valid or invalid
        validity_per_sample = wmc_per_sample
        total_valid = validity_per_sample.astype(int).sum()

        # validity over the whole batch
        validity = np.mean(wmc_per_sample)

        # indices of samples that respect all constraints (perfects)
        valid_indices = np.reshape(np.nonzero(validity_per_sample), [-1])
        valid_samples = np.take(eval_samples, valid_indices, axis=0)

        # gotta reshape to 2d to perform the uniqueness and novelty
        total_variables = reduce(lambda x, y: x * y, valid_samples.shape[1:])
        valid_samples = np.reshape(valid_samples, [-1, total_variables])

        print("Computing uniqueness")
        total_unique, uniqueness = ReachabilityMetrics.uniqueness(valid_samples)
        print("Computing novelty")
        total_novel, novelty = ReachabilityMetrics.novelty(valid_samples, training_samples)

        return (
            validity,
            total_valid,
            uniqueness,
            total_unique,
            novelty,
            total_novel
        )

    @staticmethod
    def unique_2d(x):
        """
        :return: numpy array containing the unique elements of x on the last axis.
        """
        return np.unique(x, axis=1)

    @staticmethod
    def count_in(x, y):
        """
        Assumes x and y are 2 dimensional, ~ [batch, -1].
        Assumes y elements (axis 1) are unique.
        :param x: 2d array for which we want to know how many elements are in y, each element must be unique.
        :param y: 2d array, each element must be unique.
        :return: A number showing how many elements of x which are in y.
        """
        x_shape = x.shape
        y_shape = y.shape

        # replicate tensors so that we can compare 1-1 each x element vs each y element
        x_replicated = np.tile(x, (1, y_shape[0]))
        y_replicated = np.tile(y, (x_shape[0], 1))

        # reshape replicated elements
        x_replicated = np.reshape(x_replicated, [x_shape[0] * y_shape[0], x_shape[1]])
        y_replicated = np.reshape(y_replicated, [y_shape[0] * x_shape[0], y_shape[1]])

        # element wise equality, followed by equality on all elements
        equal = (x_replicated == y_replicated)
        # x samples which have true on all columns are samples already in y
        equal_all_dims = equal.all(axis=1)
        # summing all the True values
        total_already_in = equal_all_dims.astype(float).sum()
        return total_already_in

    @staticmethod
    def novelty(x, data):
        """
        Returns the total novel items (x not in data) and the novelty (total x not in data / total x).

        :param x: array of 2d shape, [batch, -1], each element on axis 0 is a sample.
        :param data: array of 2d shape, [batch, -1], each element on axis 0 is a sample.
        :return: total number of novel items and novelty
        """
        if len(x) > 0:
            total_x = x.shape[0]

            # count the number of x that are in data too
            total_x_in_y = ReachabilityMetrics.count_in(x, data)

            # number of new samples
            count_not_x_in_y = int(total_x - total_x_in_y)

            # novelty as percentage of unseen samples
            novelty = count_not_x_in_y / total_x if total_x else 0
            return count_not_x_in_y, novelty
        else:
            return 0, 0

    @staticmethod
    def uniqueness(x):
        """
        Returns the total number of unique items and uniqueness (unique items / total items).

        :param x: array of 2d shape, [batch, -1], each element on axis 0 is a sample.
        :return: number of unique samples and uniqueness
        """
        if len(x) > 0:
            # array with unique samples
            unique_valid_samples = ReachabilityMetrics.unique_2d(x)

            total_unique = unique_valid_samples.shape[0]
            total_valid = x.shape[0]

            uniqueness = total_unique / total_valid if total_valid > 0 else 0

            return total_unique, uniqueness
        else:
            return 0, 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_samples", type=str, required=True,
                        help="Path to the input samples")
    parser.add_argument("-t", "--training_samples", type=str, required=True,
                        help="Path to the training data")
    parser.add_argument("-c", "--constraint", type=str, required=True,
                        help="The constraint that should be used", choices=['A*', 'reachability'])
    parser.add_argument("--results", type=str, required=False, default=None,
                        help="The already computed constraints")
    parser.add_argument("-s", "--settings", type=str, required=False,
                        help="Path to the mario settings json file")

    args = parser.parse_args()

    samples_folder = args.input_samples
    training_folder = args.training_samples

    # using A* constriant
    if args.constraint == 'A*':
        assert args.results is not None, "A* requires --results file as argument"
        constraint = AstarConstraint(args.results)
    # using Reachability constraint
    elif args.constraint == 'reachability':
        assert os.path.exists(args.settings), "Should specify a correct path to the mario_settings.json file"
        constraint = ReachabilityConstraint(args.settings)
    else:
        raise NotImplementedError()

    if not os.path.exists(samples_folder) or not os.path.exists(training_folder):
        print("Should specify correct samples path and training path")
    
    metrics = ReachabilityMetrics(constraint)
    validity, total_valid, uniqueness, total_unique, novelty, total_novel = \
        metrics._evaluator(samples_folder, training_folder)

    print("Results:")
    print("Validity: {0:.2f}%".format(validity*100))
    print("Valid samples: {}".format(total_valid))
    print("Uniqueness: {0:.2f}%".format(uniqueness*100))
    print("Unique samples: {}".format(total_unique))
    print("Novely: {0:.2f}%".format(novelty*100))
    print("Novel samples: {}".format(total_novel))
    print("Evaluation done!")
