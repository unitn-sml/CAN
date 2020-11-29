from functools import reduce
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori
import scipy
from scipy.spatial.distance import cdist
from scipy.ndimage import convolve

from computables import ConstraintsComputable, Computable


class Itemsets(ConstraintsComputable):
    """
    Gets some frequent itemsets and uses their presence (or lack of) in fake and real samples
    as features/constraints.
    """

    def __init__(self, experiment, training_data, validation_data, test_data, graph_nodes):
        experiment["LOGGER"].info("Computing frequent itemsets")
        data = experiment.training_data
        data = np.reshape(data, [data.shape[0], -1])
        df = pd.DataFrame(data, dtype=np.bool)
        sets = apriori(df, min_support=0.15, verbose=True)
        sets = sets[sets["itemsets"].map(len) >= 5]
        sets = sets["itemsets"].tolist()
        maxlen = max([len(s) for s in sets])
        indexes = []
        for s in sets:
            s = list(s)
            while len(s) != maxlen:
                s.append(s[0])
            indexes.append(np.array(s, ))
        self.indexes = np.concatenate(indexes, axis=0)
        self.num_sets = len(sets)
        self.equality_req = maxlen

        super().__init__(experiment, training_data, validation_data, test_data, graph_nodes)

    @staticmethod
    def constraints_names():
        """
        Returns a list of names, where ith name represent the ith constraint computed
        by the constraint function of this class. Make sure the length of this list
        is equal to the number of constraints computed for each data point.
        :return: the list of names
        """
        # just a work around, will need fixing later
        names = ["set(%s)" % s for s in range(377)]
        return names

    def _constraints_function(self, data):
        """
        This is the function that effectively computes constraints on given data, must be implemented by the user.
        :param data: Data on which to compute the constraints function.
        :return:
        """
        data = np.reshape(data, [data.shape[0], -1])
        vars = data[:, self.indexes]
        vars = np.reshape(vars, (data.shape[0], self.num_sets, self.equality_req))
        vars = np.sum(vars, axis=2)
        features = np.equal(vars, self.equality_req).astype(np.float32)
        return features


class DiscretizeNoise(Computable):

    def __init__(self, experiment, training_data, validation_data, test_data, graph_nodes):
        super().__init__(experiment, training_data, validation_data, test_data, graph_nodes)

        self.needed_for_evaluation_images = True
        assert "z" in graph_nodes
        assert "Dz" in graph_nodes
        noisedim = self.experiment["Z_DIM"]
        self.landmarks = np.random.rand(noisedim, 377)

    def _exp_kernel(self, x1, x2, l=1):
        # this is an NxD matrix, where N is number of items and D its dimensionalites
        pairwise_sq_dists = cdist(x1, x2, 'sqeuclidean')
        kernel = scipy.exp(-pairwise_sq_dists / l ** 2)
        return kernel

    def _one_hot_noise(self, z):
        matmul = np.matmul(z, self.landmarks)
        row_maxes = matmul.max(axis=1)
        row_maxes = row_maxes.reshape(-1, 1)
        matmul[:] = np.where(matmul == row_maxes, 1., 0.)
        return matmul

    def _one_hot_noise_backup(self, z):
        kernel = self._exp_kernel(z, self.landmarks)
        row_maxes = kernel.max(axis=1)
        row_maxes = row_maxes.reshape(-1, 1)
        kernel[:] = np.where(kernel == row_maxes, 1., 0.)
        return kernel

    def compute(self, feed_dict, shared_dict, curr_epoch=0, real_data_indices=None, generator_step=False,
                step_type="training"):
        onehot = self._one_hot_noise(feed_dict[self.graph_nodes["z"]])
        feed_dict[self.graph_nodes["Dz"]] = onehot
        return onehot


class DiscretizeRandom(Computable):

    def __init__(self, experiment, training_data, validation_data, test_data, graph_nodes):
        super().__init__(experiment, training_data, validation_data, test_data, graph_nodes)

        self.needed_for_evaluation_images = True
        assert "Dz" in graph_nodes

    def compute(self, feed_dict, shared_dict, curr_epoch=0, real_data_indices=None, generator_step=False,
                step_type="training"):
        choice = np.random.choice([0., 1.], 377 * self.experiment["BATCH_SIZE"], p=[0.98, 0.02])
        choice = np.reshape(choice, [self.experiment["BATCH_SIZE"], 377])
        feed_dict[self.graph_nodes["Dz"]] = choice


class StillLife1Feature(ConstraintsComputable):
    """
    This only provides the percentage of respected constraints as feature, for each data samples, so given
    data with shape [batch, SHAPE], the output will be [batch, 1].
    """

    def __init__(self, experiment, training_data, validation_data, test_data, graph_nodes):
        """
        Need to init some stuff first because its needed to compute constraints,
        the ConstraintsComputable __init__ computes some constraints, so stuff must be there
        before the super __init__.
        """
        # used later for convolution
        self.kernel = np.array(np.array([[1, 1, 1],
                                         [1, 0, 1],
                                         [1, 1, 1]]))
        # self.kernel = np.reshape(self.kernel, [3,3])
        # needed for normalization
        self.total_variables = reduce(lambda x, y: x * y, experiment["SHAPE"])

        super().__init__(experiment, training_data, validation_data, test_data, graph_nodes)

    @staticmethod
    def constraints_names():
        names = ["still_life"]
        return names

    def _constraints_function(self, data):
        res = self._still_life(data)
        return res

    def _still_life(self, data):
        # discretize just in case it is not already discretized
        discr_data = (data > 0.5).astype(np.int32)

        # perform count of neighbours (need to check if this can be done batch wise)
        discr_data = np.split(discr_data, indices_or_sections=data.shape[0], axis=0)
        constraints_not_respected = []
        for sample in discr_data:
            # needed to have the same number of dimensions as the kernel
            sample = np.squeeze(sample)

            # compute sum of neighbours for each variable
            sum_neighbours = convolve(sample, self.kernel, mode='constant')

            # sum of neighbours of elements which are zeros
            zeros_sum = sum_neighbours[sample == 0]

            # sum of neighbours of elements which are ones
            ones_sum = sum_neighbours[sample == 1]

            # check how many constraints are not being respected
            # zeros should have neighbours count != from 3
            zeros_sum = np.sum(zeros_sum == 3)
            # ones should have neighbours count either 2 or 3
            ones_sum = np.sum((ones_sum != 2) & (ones_sum != 3))
            # append normalized (percentage of not respected constraints)
            constraints_not_respected.append((zeros_sum + ones_sum) / self.total_variables)
        constraints_not_respected = np.reshape(np.stack(constraints_not_respected), [data.shape[0], 1])
        return constraints_not_respected


class StillLifeNFeatures(ConstraintsComputable):
    """
    This provides, for each variable in the data (total variables given the SHAPE), if the still life constraint
    was respected or not.
    Given input data with shape [batch, SHAPE], the output will have  shape [batch, -1].
    """

    def __init__(self, experiment, training_data, validation_data, test_data, graph_nodes):
        """
        Need to init some stuff first because its needed to compute constraints,
        the ConstraintsComputable __init__ computes some constraints, so stuff must be there
        before the super __init__.
        """
        # used later for convolution
        self.kernel = np.array(np.array([[1, 1, 1],
                                         [1, 0, 1],
                                         [1, 1, 1]]))
        # self.kernel = np.reshape(self.kernel, [3,3])
        # needed for normalization
        self.total_variables = reduce(lambda x, y: x * y, experiment["SHAPE"])

        super().__init__(experiment, training_data, validation_data, test_data, graph_nodes)

    @staticmethod
    def constraints_names():
        # dirty, i know, eventually the constraints_names should just be made a non static method
        constraints_name = ["still_life_%s" % i for i in range(784)]
        return constraints_name

    def _constraints_function(self, data):
        res = self._still_life(data)
        return res

    def _still_life(self, data):
        # discretize just in case it is not already discretized
        discr_data = (data > 0.5).astype(np.int32)

        # perform count of neighbours (need to check if this can be done batch wise)
        discr_data = np.split(discr_data, indices_or_sections=data.shape[0], axis=0)
        constraints_not_respected = []
        for sample in discr_data:
            # needed to have the same number of dimensions as the kernel
            sample = np.squeeze(sample)

            # compute sum of neighbours for each variable
            sum_neighbours = convolve(sample, self.kernel, mode='constant')

            # zeros should have neighbours count != from 3
            not_respecting_zero = (sample == 0) & (sum_neighbours == 3)
            # ones should have neighbours count either 2 or 3
            not_respecting_one = (sample == 1) & ((sum_neighbours != 3) & (sum_neighbours != 2))

            constraints_not_respected.append(
                np.reshape((not_respecting_zero | not_respecting_one).astype(np.float32), -1))

        constraints_not_respected = np.reshape(np.stack(constraints_not_respected), [data.shape[0], -1])
        return constraints_not_respected
