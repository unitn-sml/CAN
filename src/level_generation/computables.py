import numpy as np
from numpy import empty as np_empty, float32 as np_float32, sum as np_sum
import tensorflow as tf

from computables import Computable, ConstraintsComputable

class Reachability(ConstraintsComputable):
    def __init__(self, experiment, training_data, validation_data, test_data, graph_nodes):
        super().__init__(experiment, training_data, validation_data, test_data, graph_nodes)

        self.kernel = np.asarray([[1, 1, 0], [1, 1, 0], [1, 1, 0]])
        self.paddings_down = np.array([[0, 0], [0, 1], [0, 0]])
        self.paddings_up = np.array([[0, 0], [1, 0], [0, 0]])

        self.kernel_height = self.kernel.shape[0]
        self.kernel_width = self.kernel.shape[1]

        self.conv_paddings = np.array([
            [0, 0],
            [(self.kernel_height - 1) // 2, (self.kernel_height - 1) // 2],
            [(self.kernel_width - 1) // 2, (self.kernel_width - 1) // 2]
        ])

    def simple_convolution(self, data):
        res = np.zeros(data.shape)
        data_height = data.shape[1]
        data_width = data.shape[2]

        data = np.pad(data, self.conv_paddings, mode='constant', constant_values=0)
        ker = np.tile(self.kernel, [data.shape[0], 1, 1])

        for i in range(data_height):
            for j in range(data_width):
                res[:, i, j] = np.sum(
                    np.multiply(data[:, i:i+self.kernel_height, j:j+self.kernel_width], ker), axis=(1, 2)
                )
        return res

    @staticmethod
    def constraints_names():
        """
        Returns a list of names, where ith name represent the ith constraint computed
        by the constraint function of this class. Make sure the length of this list
        is equal to the number of constraints computed for each data point.
        :return: the list of names
        """
        names = [
            "reachability"
        ]
        return names

    def _constraints_function(self, data):
        """
        This is the function that effectively computes constraints on given data, must be implemented by the user.
        :param data: Data on which to compute the constraints function.
        :return:
        """
        # data.shape[-3:] -> (height, width, channels)
        level = np.reshape(data, [-1] + data.shape[-3:])
        air_tiles = [2, 5]
        level = sum([level == x for x in air_tiles])

        level_height = level.shape[1]
        level_width = level.shape[2]

        # create matrix of probabilities to be a stationary point
        shadow = np.pad(1 - level[:, 1:, :], self.paddings_down, mode='constant', constant_values=0)
        stationary = np.multiply(level, shadow)

        # use the convolution to find all reachables points from the stationary points
        rechable = simple_convolution(stationary, kernel)

        # values to interval 0 - 1
        rechable = np.clip(rechable, 0, 1)

        # lower values where air is less probable
        rechable = np.multiply(rechable, level)

        # the fall of big values. simulate possible falls to the bottom through various air-blocks
        new_list = [rechable[:, 0:1, :]]

        for i in range(1, level_height):
            new_row = np.multiply(new_list[i-1], level[:, i:i+1, :])
            new_row = np.maximum(new_row, rechable[:, i:i+1, :])
            new_list.append(new_row)

        rechable = np.concatenate(new_list, axis=1)

        # start from some block on the first column
        new_array = [rechable[:, :, 0:1]]

        for i in range(1, level_width):
            previous_rechable_column = new_array[i-1]
            actual_rechable_column = rechable[:, :, i:i+1]
            new_column = np.multiply(previous_rechable_column, actual_rechable_column)

            # jumps
            for j in range(level_height - 1):
                mult = np.pad(new_column[:, :-1, :], self.paddings_up, mode='constant', constant_values=.0)
                movement = np.multiply(mult, actual_rechable_column)
                new_column = np.maximum(new_column, movement)
            
            # falls
            for j in range(level_height - 1):
                mult = np.pad(new_column[:, 1:, :], self.paddings_down, mode='constant', constant_values=.0)
                movement = np.multiply(mult, actual_rechable_column)
                new_column = np.maximum(new_column, movement)

            new_array.append(new_column)

        rechable = np.concatenate(new_array, axis=2)

        return np.concatenate([np.amax(rechable[:, :, -1], axis=-1)], axis=1)

        



