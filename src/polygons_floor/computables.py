import numpy as np
from numpy import empty as np_empty, float32 as np_float32, sum as np_sum
import tensorflow as tf

from computables import Computable, ConstraintsComputable


class RandomUniform(Computable):
    """
    Computable to feed random uniform noise [0,1) in the feed dictionary. Requires
    the presence of the "R" placeholder in the graph_nodes dict, that node will be mapped
    to random uniform noise in the feed_dict dictionary by calling the compute method.
    You should use/add this before other computables, given/if the output of your generator depends
    on this noise.
    """

    def __init__(self, experiment, training_data, validation_data, test_data, graph_nodes):
        super().__init__(experiment, training_data, validation_data, test_data, graph_nodes)

        assert "R" in graph_nodes, "Expected to find the R placeholder, used for adding random uniform noise to the " \
                                   "computation graph. "
        self.random_uniform_noise_generator = tf.random_uniform(graph_nodes["R"].shape)
        self.needed_for_evaluation_images = True

    def compute(self, feed_dict, shared_dict, curr_epoch=0, real_data_indices=None, generator_step=False,
                step_type="training"):
        feed_dict[self.graph_nodes["R"]] = self.random_uniform_noise_generator.eval()


class AreaAndParityCheck20(ConstraintsComputable):
    def __init__(self, experiment, training_data, validation_data, test_data, graph_nodes):
        super().__init__(experiment, training_data, validation_data, test_data, graph_nodes)

    def _constraints_function(self, data):
        """
        This is the function that effectively computes constraints on given data, must be implemented by the user.
        :param data: Data on which to compute the constraints function.
        :return:
        """
        area = self.experiment["AREA"]
        polygons_number = self.experiment["POLYGONS_NUMBER"]
        shape = self.experiment["SHAPE"]
        data = np.expand_dims(data, 0)
        res = self._area_and_parity_check_fake_data(data, area, polygons_number, shape)
        res = np.squeeze(res, 0)
        return res

    def _area_and_parity_check(self, data_mb, area, polygons_number, shape):
        img_width = shape[0]
        img_height = shape[1]

        target_area = area * polygons_number
        nonzero = np_sum(data_mb[:, 1:-1, 1:-1], axis=(1, 2))
        norm = img_width * img_height - nonzero

        greater_area_inner = np.minimum(1, np.maximum(0, nonzero - target_area) / norm)
        smaller_area_inner = np.minimum(1, np.maximum(0, target_area - nonzero) / norm)

        sum_rows = np_sum(data_mb[:, 1:int(img_width / 2), 1:int(img_width / 2), 0], 2)
        check_rows = np.mod(sum_rows, 2)
        check_rows = np.not_equal(data_mb[:, 1:int(img_width / 2), 0, 0], check_rows).astype(float)

        sum_cols = np_sum(data_mb[:, 1:int(img_height / 2), 1:int(img_height / 2), 0], 1)
        check_cols = np.mod(sum_cols, 2)
        check_cols = np.not_equal(data_mb[:, 0, 1:int(img_height / 2), 0], check_cols).astype(float)

        mb_constraints_values = np.concatenate([greater_area_inner, smaller_area_inner, check_rows, check_cols], axis=1)

        return mb_constraints_values


class ParityCheck20(ConstraintsComputable):
    def __init__(self, experiment, training_data, validation_data, test_data, graph_nodes):
        super().__init__(experiment, training_data, validation_data, test_data, graph_nodes)

    @staticmethod
    def constraints_names():
        """
        Returns a list of names, where ith name represent the ith constraint computed
        by the constraint function of this class. Make sure the length of this list
        is equal to the number of constraints computed for each data point.
        :return: the list of names
        """
        names = [
            "_parity_check_rl_1",
            "_parity_check_rl_2",
            "_parity_check_rl_3",
            "_parity_check_rl_4",
            "_parity_check_rl_5",
            "_parity_check_rl_6",
            "_parity_check_rl_7",
            "_parity_check_rl_8",
            "_parity_check_rl_9",
            "_parity_check_ct_1",
            "_parity_check_ct_2",
            "_parity_check_ct_3",
            "_parity_check_ct_4",
            "_parity_check_ct_5",
            "_parity_check_ct_6",
            "_parity_check_ct_7",
            "_parity_check_ct_8",
            "_parity_check_ct_9"]
        return names

    def _constraints_function(self, data):
        """
        This is the function that effectively computes constraints on given data, must be implemented by the user.
        :param data: Data on which to compute the constraints function.
        :return:
        """
        area = self.experiment["AREA"]
        polygons_number = self.experiment["POLYGONS_NUMBER"]
        shape = self.experiment["SHAPE"]
        data = np.expand_dims(data, 0)
        res = self._area_and_parity_check_fake_data(data, area, polygons_number, shape)
        res = np.squeeze(res, 0)
        return res

    def _area_and_parity_check(self, data_mb, area, polygons_number, shape):
        img_width = shape[0]
        img_height = shape[1]

        sum_rows = np_sum(data_mb[:, 1:int(img_width / 2), 1:int(img_width / 2), 0], 2)
        check_rows = np.mod(sum_rows, 2)
        check_rows = np.not_equal(data_mb[:, 1:int(img_width / 2), 0, 0], check_rows).astype(float)

        sum_cols = np_sum(data_mb[:, 1:int(img_height / 2), 1:int(img_height / 2), 0], 1)
        check_cols = np.mod(sum_cols, 2)
        check_cols = np.not_equal(data_mb[:, 0, 1:int(img_height / 2), 0], check_cols).astype(float)

        mb_constraints_values = np.concatenate([check_rows, check_cols], axis=1)

        return mb_constraints_values

    def _area_and_parity_check_fake_data(self, data, area, polygons_number, shape):
        # we have (n_samples x bs) generated items; for each of them we have to
        # compute the values of each constraint.
        # R_mb has shape [n_samples, bs] + image_shape
        # e.g. [n_samples, bs, 20, 20, 1]
        # so we need to iterate over the first two dimensions

        img_width = shape[0]
        img_height = shape[1]

        # check parity of stuff after first pixel and to half of the image
        sum_rows = np_sum(data[:, :, 1:int(img_width / 2), 1:int(img_width / 2), 0], 3)
        check_rows = np.mod(sum_rows, 2)
        # check if the first pixel of the row is 1 if the sum of the pixel to its right up to half the row
        # its odd
        check_rows = np.not_equal(data[:, :, 1:int(img_width / 2), 0, 0], check_rows).astype(float)

        sum_cols = np_sum(data[:, :, 1:int(img_height / 2), 1:int(img_height / 2), 0], 2)
        check_cols = np.mod(sum_cols, 2)
        check_cols = np.not_equal(data[:, :, 0, 1:int(img_height / 2), 0], check_cols).astype(float)

        return np.concatenate([check_rows, check_cols], axis=2)


class ParityCheck20_full(ConstraintsComputable):
    def __init__(self, experiment, training_data, validation_data, test_data, graph_nodes):
        super().__init__(experiment, training_data, validation_data, test_data, graph_nodes)

    @staticmethod
    def constraints_names():
        """
        Returns a list of names, where ith name represent the ith constraint computed
        by the constraint function of this class. Make sure the length of this list
        is equal to the number of constraints computed for each data point.
        :return: the list of names
        """
        names = ["_parity_check_%s" % i for i in range(72)]
        return names

    def _constraints_function(self, data):
        """
        This is the function that effectively computes constraints on given data, must be implemented by the user.
        :param data: Data on which to compute the constraints function.
        :return:
        """
        area = self.experiment["AREA"]
        polygons_number = self.experiment["POLYGONS_NUMBER"]
        shape = self.experiment["SHAPE"]
        data = np.expand_dims(data, 0)
        res = self._area_and_parity_check_fake_data(data, area, polygons_number, shape)
        res = np.squeeze(res, 0)
        return res

    def _area_and_parity_check_fake_data(self, data, area, polygons_number, shape):
        img_width = shape[0]
        img_height = shape[1]

        sum_rows_l = np_sum(data[:, :, 1:(img_width - 1), 1:int(img_width / 2), 0], 3)
        check_rows_l = np.mod(sum_rows_l, 2)
        check_rows_l = np.not_equal(data[:, :, 1:(img_width - 1), 0, 0], check_rows_l).astype(float)

        sum_rows_r = np_sum(data[:, :, 1:(img_width - 1), int(img_width / 2):(img_width - 1), 0], 3)
        check_rows_r = np.mod(sum_rows_r, 2)
        check_rows_r = np.not_equal(data[:, :, 1:(img_width - 1), img_width - 1, 0], check_rows_r).astype(float)

        sum_cols_top = np_sum(data[:, :, 1:int(img_height / 2), 1:(img_height - 1), 0], 2)
        check_cols_top = np.mod(sum_cols_top, 2)
        check_cols_top = np.not_equal(data[:, :, 0, 1:(img_height - 1), 0], check_cols_top).astype(float)

        sum_cols_bot = np_sum(data[:, :, int(img_height / 2): (img_height - 1), 1:(img_height - 1), 0], 2)
        check_cols_bot = np.mod(sum_cols_bot, 2)
        check_cols_bot = np.not_equal(data[:, :, img_height - 1, 1:(img_height - 1), 0], check_cols_bot).astype(float)

        return np.concatenate([check_rows_l, check_rows_r, check_cols_top, check_cols_bot], axis=2)


class ParityCheck20_1sided(ConstraintsComputable):
    def __init__(self, experiment, training_data, validation_data, test_data, graph_nodes):
        super().__init__(experiment, training_data, validation_data, test_data, graph_nodes)

    def _constraints_function(self, data):
        """
        This is the function that effectively computes constraints on given data, must be implemented by the user.
        :param data: Data on which to compute the constraints function.
        :return:
        """
        area = self.experiment["AREA"]
        polygons_number = self.experiment["POLYGONS_NUMBER"]
        shape = self.experiment["SHAPE"]
        data = np.expand_dims(data, 0)
        res = self._area_and_parity_check_fake_data(data, area, polygons_number, shape)
        res = np.squeeze(res, 0)
        return res

    def _area_and_parity_check_fake_data(self, data, area, polygons_number, shape):
        # we have (n_samples x bs) generated items; for each of them we have to
        # compute the values of each constraint.
        # R_mb has shape [n_samples, bs] + image_shape
        # e.g. [n_samples, bs, 20, 20, 1]
        # so we need to iterate over the first two dimensions

        img_width = shape[0]
        img_height = shape[1]

        # check parity of stuff after first pixel and to half of the image
        sum_rows = np_sum(data[:, :, 1:int(img_width / 2), 1:int(img_width / 2), 0], 3)
        check_rows = np.mod(sum_rows, 2)
        # check if the first pixel of the row is 1 if the sum of the pixel to its right up to half the row
        # its odd
        check_rows = np.not_equal(data[:, :, 1:int(img_width / 2), 0, 0], check_rows).astype(bool) & check_rows.astype(
            bool)
        check_rows = check_rows.astype(float)

        sum_cols = np_sum(data[:, :, 1:int(img_height / 2), 1:int(img_height / 2), 0], 2)
        check_cols = np.mod(sum_cols, 2)
        check_cols = np.not_equal(data[:, :, 0, 1:int(img_height / 2), 0], check_cols).astype(
            bool) & check_cols.astype(bool)
        check_cols = check_cols.astype(float)

        return np.concatenate([check_rows, check_cols], axis=2)

    @staticmethod
    def constraints_names():
        """
        Returns a list of names, where ith name represent the ith constraint computed
        by the constraint function of this class. Make sure the length of this list
        is equal to the number of constraints computed for each data point.
        :return: the list of names
        """
        names = [
            "_parity_check_rl_1",
            "_parity_check_rl_2",
            "_parity_check_rl_3",
            "_parity_check_rl_4",
            "_parity_check_rl_5",
            "_parity_check_rl_6",
            "_parity_check_rl_7",
            "_parity_check_rl_8",
            "_parity_check_rl_9",
            "_parity_check_ct_1",
            "_parity_check_ct_2",
            "_parity_check_ct_3",
            "_parity_check_ct_4",
            "_parity_check_ct_5",
            "_parity_check_ct_6",
            "_parity_check_ct_7",
            "_parity_check_ct_8",
            "_parity_check_ct_9"]
        return names


class AreaAndAllParityCheckMulti20(ConstraintsComputable):
    def __init__(self, experiment, training_data, validation_data, test_data, graph_nodes):
        super().__init__(experiment, training_data, validation_data, test_data, graph_nodes)

    @staticmethod
    def constraints_names():
        """
        Returns a list of names, where ith name represent the ith constraint computed
        by the constraint function of this class. Make sure the length of this list
        is equal to the number of constraints computed for each data point.
        :return:
        """
        names = [
            "_greater_white_area_inner",
            "_smaller_white_area_inner",
            "_greater_red_area_inner",
            "_smaller_red_area_inner",
            "_parity_check_rl_1",
            "_parity_check_rl_2",
            "_parity_check_rl_3",
            "_parity_check_rl_4",
            "_parity_check_rl_5",
            "_parity_check_rl_6",
            "_parity_check_rl_7",
            "_parity_check_rl_8",
            "_parity_check_rl_9",
            "_parity_check_rl_10",
            "_parity_check_rl_11",
            "_parity_check_rl_12",
            "_parity_check_rl_13",
            "_parity_check_rl_14",
            "_parity_check_rl_15",
            "_parity_check_rl_16",
            "_parity_check_rl_17",
            "_parity_check_rl_18",
            "_parity_check_ct_1",
            "_parity_check_ct_2",
            "_parity_check_ct_3",
            "_parity_check_ct_4",
            "_parity_check_ct_5",
            "_parity_check_ct_6",
            "_parity_check_ct_7",
            "_parity_check_ct_8",
            "_parity_check_ct_9",
            "_parity_check_ct_10",
            "_parity_check_ct_11",
            "_parity_check_ct_12",
            "_parity_check_ct_13",
            "_parity_check_ct_14",
            "_parity_check_ct_15",
            "_parity_check_ct_16",
            "_parity_check_ct_17",
            "_parity_check_ct_18",
            "_parity_check_rr_1",
            "_parity_check_rr_2",
            "_parity_check_rr_3",
            "_parity_check_rr_4",
            "_parity_check_rr_5",
            "_parity_check_rr_6",
            "_parity_check_rr_7",
            "_parity_check_rr_8",
            "_parity_check_rr_9",
            "_parity_check_rr_10",
            "_parity_check_rr_11",
            "_parity_check_rr_12",
            "_parity_check_rr_13",
            "_parity_check_rr_14",
            "_parity_check_rr_15",
            "_parity_check_rr_16",
            "_parity_check_rr_17",
            "_parity_check_rr_18",
            "_parity_check_cb_1",
            "_parity_check_cb_2",
            "_parity_check_cb_3",
            "_parity_check_cb_4",
            "_parity_check_cb_5",
            "_parity_check_cb_6",
            "_parity_check_cb_7",
            "_parity_check_cb_8",
            "_parity_check_cb_9",
            "_parity_check_cb_10",
            "_parity_check_cb_11",
            "_parity_check_cb_12",
            "_parity_check_cb_13",
            "_parity_check_cb_14",
            "_parity_check_cb_15",
            "_parity_check_cb_16",
            "_parity_check_cb_17",
            "_parity_check_cb_18"]
        return names

    def _constraints_function(self, data):
        """
        This is the function that effectively computes constraints on given data, must be implemented by the user.
        :param data: Data on which to compute the constraints function.
        :return:
        """
        area = self.experiment["AREA"]
        polygons_number = self.experiment["POLYGONS_NUMBER"]
        shape = self.experiment["SHAPE"]
        color_map = self.experiment["COLOR_MAP"]
        return self._area_and_all_parity_check_real_data_multi(data, area, polygons_number, shape, color_map)

    def _area_and_all_parity_check_real_data_multi(self, data_mb, area, polygons_number, shape, color_map):
        # we have (n_samples x bs) generated items; for each of them we have to
        # compute the values of each constraint.
        # R_mb has shape [n_samples, bs] + image_shape
        # e.g. [n_samples, bs, 20, 20, 1]
        # so we need to iterate over the first two dimensions
        img_width = shape[0]
        img_height = shape[1]

        black = (0, 0, 0)
        white = (255, 255, 255)
        red = (255, 0, 0)
        green = (0, 255, 0)
        blue = (0, 0, 255)
        parity_colors = [black, white, red, green, blue]

        white_index = color_map[white]
        red_index = color_map[red]
        black_index = color_map[black]
        white_area = area
        red_area = area

        white_pixel = np_sum(data_mb[:, 1:-1, 1:-1, white_index] == 1, axis=(1, 2))
        white_pixel = white_pixel.reshape(white_pixel.shape + (1,))

        red_pixel = np_sum(data_mb[:, 1:-1, 1:-1, red_index] == 1, axis=(1, 2))
        red_pixel = red_pixel.reshape(red_pixel.shape + (1,))

        norm = img_width * img_height - white_area
        greater_white_area_inner = \
            np.minimum(1, np.maximum(0, white_pixel - white_area) / norm)
        smaller_white_area_inner = \
            np.minimum(1, np.maximum(0, white_area - white_pixel) / norm)

        norm = img_width * img_height - red_area
        greater_red_area_inner = \
            np.minimum(1, np.maximum(0, red_pixel - red_area) / norm)
        smaller_red_area_inner = \
            np.minimum(1, np.maximum(0, red_area - red_pixel) / norm)

        sum_left = np_sum(
            data_mb[:, 1:(img_width - 1), 1:int(img_width / 2), black_index] == 0, 2)
        check_left = np.mod(sum_left, len(parity_colors))

        parity_index_left = np.argmax(data_mb[:, 1:(img_width - 1), 0, :], axis=-1)
        check_left = np.abs(parity_index_left - check_left) / (len(parity_colors) - 1)

        sum_top = np_sum(
            data_mb[:, 1:int(img_height / 2), 1:(img_height - 1), black_index] == 0, 1)
        check_top = np.mod(sum_top, len(parity_colors))
        parity_index_top = np.argmax(data_mb[:, 0, 1:(img_height - 1), :], axis=-1)
        check_top = np.abs(parity_index_top - check_top) / (len(parity_colors) - 1)

        sum_right = np_sum(
            data_mb[:, 1:(img_width - 1), int(img_width / 2):(img_width - 1),
            black_index] == 0, 2)
        check_right = np.mod(sum_right, len(parity_colors))
        parity_index_right = \
            np.argmax(data_mb[:, 1:(img_width - 1), img_width - 1, :], axis=-1)
        check_right = \
            np.abs(parity_index_right - check_right) / (len(parity_colors) - 1)

        sum_bottom = np_sum(
            data_mb[:, int(img_height / 2):(img_height - 1), 1:(img_height - 1),
            black_index] == 0, 1)
        check_bottom = np.mod(sum_bottom, len(parity_colors))
        parity_index_bottom = np.argmax(
            data_mb[:, (img_height - 1), 1:(img_height - 1), :], axis=-1)
        check_bottom = \
            np.abs(parity_index_bottom - check_bottom) / (len(parity_colors) - 1)

        mb_constraints_values = \
            np.concatenate([greater_white_area_inner, smaller_white_area_inner,
                            greater_red_area_inner, smaller_red_area_inner,
                            check_left, check_top, check_right, check_bottom],
                           axis=1)

        return mb_constraints_values


class AreaAndAllParityCheckMulti32(ConstraintsComputable):
    def __init__(self, experiment, training_data, validation_data, test_data, graph_nodes):
        super().__init__(experiment, training_data, validation_data, test_data, graph_nodes)

    @staticmethod
    def constraints_names():
        """
        Returns a list of names, where ith name represent the ith constraint computed
        by the constraint function of this class. Make sure the length of this list
        is equal to the number of constraints computed for each data point.
        :return:
        """
        names = [
            "_greater_area_inner",
            "_smaller_area_inner",
            "_parity_check_rl_1",
            "_parity_check_rl_2",
            "_parity_check_rl_3",
            "_parity_check_rl_4",
            "_parity_check_rl_5",
            "_parity_check_rl_6",
            "_parity_check_rl_7",
            "_parity_check_rl_8",
            "_parity_check_rl_9",
            "_parity_check_rl_10",
            "_parity_check_rl_11",
            "_parity_check_rl_12",
            "_parity_check_rl_13",
            "_parity_check_rl_14",
            "_parity_check_rl_15",
            "_parity_check_rl_16",
            "_parity_check_rl_17",
            "_parity_check_rl_18",
            "_parity_check_rl_19",
            "_parity_check_rl_20",
            "_parity_check_rl_21",
            "_parity_check_rl_22",
            "_parity_check_rl_23",
            "_parity_check_rl_24",
            "_parity_check_rl_25",
            "_parity_check_rl_26",
            "_parity_check_rl_27",
            "_parity_check_rl_28",
            "_parity_check_rl_29",
            "_parity_check_rl_30",
            "_parity_check_ct_1",
            "_parity_check_ct_2",
            "_parity_check_ct_3",
            "_parity_check_ct_4",
            "_parity_check_ct_5",
            "_parity_check_ct_6",
            "_parity_check_ct_7",
            "_parity_check_ct_8",
            "_parity_check_ct_9",
            "_parity_check_ct_10",
            "_parity_check_ct_11",
            "_parity_check_ct_12",
            "_parity_check_ct_13",
            "_parity_check_ct_14",
            "_parity_check_ct_15",
            "_parity_check_ct_16",
            "_parity_check_ct_17",
            "_parity_check_ct_18",
            "_parity_check_ct_19",
            "_parity_check_ct_20",
            "_parity_check_ct_21",
            "_parity_check_ct_22",
            "_parity_check_ct_23",
            "_parity_check_ct_24",
            "_parity_check_ct_25",
            "_parity_check_ct_26",
            "_parity_check_ct_27",
            "_parity_check_ct_28",
            "_parity_check_ct_29",
            "_parity_check_ct_30",
            "_parity_check_rr_1",
            "_parity_check_rr_2",
            "_parity_check_rr_3",
            "_parity_check_rr_4",
            "_parity_check_rr_5",
            "_parity_check_rr_6",
            "_parity_check_rr_7",
            "_parity_check_rr_8",
            "_parity_check_rr_9",
            "_parity_check_rr_10",
            "_parity_check_rr_11",
            "_parity_check_rr_12",
            "_parity_check_rr_13",
            "_parity_check_rr_14",
            "_parity_check_rr_15",
            "_parity_check_rr_16",
            "_parity_check_rr_17",
            "_parity_check_rr_18",
            "_parity_check_rr_19",
            "_parity_check_rr_20",
            "_parity_check_rr_21",
            "_parity_check_rr_22",
            "_parity_check_rr_23",
            "_parity_check_rr_24",
            "_parity_check_rr_25",
            "_parity_check_rr_26",
            "_parity_check_rr_27",
            "_parity_check_rr_28",
            "_parity_check_rr_29",
            "_parity_check_rr_30",
            "_parity_check_cb_1",
            "_parity_check_cb_2",
            "_parity_check_cb_3",
            "_parity_check_cb_4",
            "_parity_check_cb_5",
            "_parity_check_cb_6",
            "_parity_check_cb_7",
            "_parity_check_cb_8",
            "_parity_check_cb_9",
            "_parity_check_cb_10",
            "_parity_check_cb_11",
            "_parity_check_cb_12",
            "_parity_check_cb_13",
            "_parity_check_cb_14",
            "_parity_check_cb_15",
            "_parity_check_cb_16",
            "_parity_check_cb_17",
            "_parity_check_cb_18",
            "_parity_check_cb_19",
            "_parity_check_cb_20",
            "_parity_check_cb_21",
            "_parity_check_cb_22",
            "_parity_check_cb_23",
            "_parity_check_cb_24",
            "_parity_check_cb_25",
            "_parity_check_cb_26",
            "_parity_check_cb_27",
            "_parity_check_cb_28",
            "_parity_check_cb_29",
            "_parity_check_cb_30"]
        return names

    def _constraints_function(self, data):
        """
        This is the function that effectively computes constraints on given data, must be implemented by the user.
        :param data: Data on which to compute the constraints function.
        :return:
        """
        area = self.experiment["AREA"]
        polygons_number = self.experiment["POLYGONS_NUMBER"]
        shape = self.experiment["SHAPE"]
        color_map = self.experiment["COLOR_MAP"]
        return self._area_and_all_parity_check_real_data_multi(data, area, polygons_number, shape, color_map)

    def _area_and_all_parity_check_real_data_multi(self, data_mb, area, polygons_number, shape, color_map):
        # we have (n_samples x bs) generated items; for each of them we have to
        # compute the values of each constraint.
        # R_mb has shape [n_samples, bs] + image_shape
        # e.g. [n_samples, bs, 20, 20, 1]
        # so we need to iterate over the first two dimensions
        img_width = shape[0]
        img_height = shape[1]

        black = (0, 0, 0)
        white = (255, 255, 255)
        red = (255, 0, 0)
        green = (0, 255, 0)
        blue = (0, 0, 255)
        parity_colors = [black, white, red, green, blue]

        white_index = color_map[white]
        red_index = color_map[red]
        black_index = color_map[black]
        white_area = area
        red_area = area

        white_pixel = np_sum(data_mb[:, 1:-1, 1:-1, white_index] == 1, axis=(1, 2))
        white_pixel = white_pixel.reshape(white_pixel.shape + (1,))

        red_pixel = np_sum(data_mb[:, 1:-1, 1:-1, red_index] == 1, axis=(1, 2))
        red_pixel = red_pixel.reshape(red_pixel.shape + (1,))

        norm = img_width * img_height - white_area
        greater_white_area_inner = \
            np.minimum(1, np.maximum(0, white_pixel - white_area) / norm)
        smaller_white_area_inner = \
            np.minimum(1, np.maximum(0, white_area - white_pixel) / norm)

        norm = img_width * img_height - red_area
        greater_red_area_inner = \
            np.minimum(1, np.maximum(0, red_pixel - red_area) / norm)
        smaller_red_area_inner = \
            np.minimum(1, np.maximum(0, red_area - red_pixel) / norm)

        sum_left = np_sum(
            data_mb[:, 1:(img_width - 1), 1:int(img_width / 2), black_index] == 0, 2)
        check_left = np.mod(sum_left, len(parity_colors))

        parity_index_left = np.argmax(data_mb[:, 1:(img_width - 1), 0, :], axis=-1)
        check_left = np.abs(parity_index_left - check_left) / (len(parity_colors) - 1)

        sum_top = np_sum(
            data_mb[:, 1:int(img_height / 2), 1:(img_height - 1), black_index] == 0, 1)
        check_top = np.mod(sum_top, len(parity_colors))
        parity_index_top = np.argmax(data_mb[:, 0, 1:(img_height - 1), :], axis=-1)
        check_top = np.abs(parity_index_top - check_top) / (len(parity_colors) - 1)

        sum_right = np_sum(
            data_mb[:, 1:(img_width - 1), int(img_width / 2):(img_width - 1),
            black_index] == 0, 2)
        check_right = np.mod(sum_right, len(parity_colors))
        parity_index_right = \
            np.argmax(data_mb[:, 1:(img_width - 1), img_width - 1, :], axis=-1)
        check_right = \
            np.abs(parity_index_right - check_right) / (len(parity_colors) - 1)

        sum_bottom = np_sum(
            data_mb[:, int(img_height / 2):(img_height - 1), 1:(img_height - 1),
            black_index] == 0, 1)
        check_bottom = np.mod(sum_bottom, len(parity_colors))
        parity_index_bottom = np.argmax(
            data_mb[:, (img_height - 1), 1:(img_height - 1), :], axis=-1)
        check_bottom = \
            np.abs(parity_index_bottom - check_bottom) / (len(parity_colors) - 1)

        mb_constraints_values = \
            np.concatenate([greater_white_area_inner, smaller_white_area_inner,
                            greater_red_area_inner, smaller_red_area_inner,
                            check_left, check_top, check_right, check_bottom],
                           axis=1)

        return mb_constraints_values


class AreaAndAllParityCheck60(ConstraintsComputable):
    def __init__(self, experiment, training_data, validation_data, test_data, graph_nodes):
        super().__init__(experiment, training_data, validation_data, test_data, graph_nodes)

    @staticmethod
    def constraints_names():
        """
        Returns a list of names, where ith name represent the ith constraint computed
        by the constraint function of this class. Make sure the length of this list
        is equal to the number of constraints computed for each data point.
        :return:
        """
        names = [
            "_greater_area_inner",
            "_smaller_area_inner",
            "_parity_check_rl_1",
            "_parity_check_rl_2",
            "_parity_check_rl_3",
            "_parity_check_rl_4",
            "_parity_check_rl_5",
            "_parity_check_rl_6",
            "_parity_check_rl_7",
            "_parity_check_rl_8",
            "_parity_check_rl_9",
            "_parity_check_rl_10",
            "_parity_check_rl_11",
            "_parity_check_rl_12",
            "_parity_check_rl_13",
            "_parity_check_rl_14",
            "_parity_check_rl_15",
            "_parity_check_rl_16",
            "_parity_check_rl_17",
            "_parity_check_rl_18",
            "_parity_check_rl_19",
            "_parity_check_rl_20",
            "_parity_check_rl_21",
            "_parity_check_rl_22",
            "_parity_check_rl_23",
            "_parity_check_rl_24",
            "_parity_check_rl_25",
            "_parity_check_rl_26",
            "_parity_check_rl_27",
            "_parity_check_rl_28",
            "_parity_check_rl_29",
            "_parity_check_rl_30",
            "_parity_check_rl_31",
            "_parity_check_rl_32",
            "_parity_check_rl_33",
            "_parity_check_rl_34",
            "_parity_check_rl_35",
            "_parity_check_rl_36",
            "_parity_check_rl_37",
            "_parity_check_rl_38",
            "_parity_check_rl_39",
            "_parity_check_rl_40",
            "_parity_check_rl_41",
            "_parity_check_rl_42",
            "_parity_check_rl_43",
            "_parity_check_rl_44",
            "_parity_check_rl_45",
            "_parity_check_rl_46",
            "_parity_check_rl_47",
            "_parity_check_rl_48",
            "_parity_check_rl_49",
            "_parity_check_rl_50",
            "_parity_check_rl_51",
            "_parity_check_rl_52",
            "_parity_check_rl_53",
            "_parity_check_rl_54",
            "_parity_check_rl_55",
            "_parity_check_rl_56",
            "_parity_check_rl_57",
            "_parity_check_rl_58",
            "_parity_check_ct_1",
            "_parity_check_ct_2",
            "_parity_check_ct_3",
            "_parity_check_ct_4",
            "_parity_check_ct_5",
            "_parity_check_ct_6",
            "_parity_check_ct_7",
            "_parity_check_ct_8",
            "_parity_check_ct_9",
            "_parity_check_ct_10",
            "_parity_check_ct_11",
            "_parity_check_ct_12",
            "_parity_check_ct_13",
            "_parity_check_ct_14",
            "_parity_check_ct_15",
            "_parity_check_ct_16",
            "_parity_check_ct_17",
            "_parity_check_ct_18",
            "_parity_check_ct_19",
            "_parity_check_ct_20",
            "_parity_check_ct_21",
            "_parity_check_ct_22",
            "_parity_check_ct_23",
            "_parity_check_ct_24",
            "_parity_check_ct_25",
            "_parity_check_ct_26",
            "_parity_check_ct_27",
            "_parity_check_ct_28",
            "_parity_check_ct_29",
            "_parity_check_ct_30",
            "_parity_check_ct_31",
            "_parity_check_ct_32",
            "_parity_check_ct_33",
            "_parity_check_ct_34",
            "_parity_check_ct_35",
            "_parity_check_ct_36",
            "_parity_check_ct_37",
            "_parity_check_ct_38",
            "_parity_check_ct_39",
            "_parity_check_ct_40",
            "_parity_check_ct_41",
            "_parity_check_ct_42",
            "_parity_check_ct_43",
            "_parity_check_ct_44",
            "_parity_check_ct_45",
            "_parity_check_ct_46",
            "_parity_check_ct_47",
            "_parity_check_ct_48",
            "_parity_check_ct_49",
            "_parity_check_ct_50",
            "_parity_check_ct_51",
            "_parity_check_ct_52",
            "_parity_check_ct_53",
            "_parity_check_ct_54",
            "_parity_check_ct_55",
            "_parity_check_ct_56",
            "_parity_check_ct_57",
            "_parity_check_ct_58",
            "_parity_check_rr_1",
            "_parity_check_rr_2",
            "_parity_check_rr_3",
            "_parity_check_rr_4",
            "_parity_check_rr_5",
            "_parity_check_rr_6",
            "_parity_check_rr_7",
            "_parity_check_rr_8",
            "_parity_check_rr_9",
            "_parity_check_rr_10",
            "_parity_check_rr_11",
            "_parity_check_rr_12",
            "_parity_check_rr_13",
            "_parity_check_rr_14",
            "_parity_check_rr_15",
            "_parity_check_rr_16",
            "_parity_check_rr_17",
            "_parity_check_rr_18",
            "_parity_check_rr_19",
            "_parity_check_rr_20",
            "_parity_check_rr_21",
            "_parity_check_rr_22",
            "_parity_check_rr_23",
            "_parity_check_rr_24",
            "_parity_check_rr_25",
            "_parity_check_rr_26",
            "_parity_check_rr_27",
            "_parity_check_rr_28",
            "_parity_check_rr_29",
            "_parity_check_rr_30",
            "_parity_check_rr_31",
            "_parity_check_rr_32",
            "_parity_check_rr_33",
            "_parity_check_rr_34",
            "_parity_check_rr_35",
            "_parity_check_rr_36",
            "_parity_check_rr_37",
            "_parity_check_rr_38",
            "_parity_check_rr_39",
            "_parity_check_rr_40",
            "_parity_check_rr_41",
            "_parity_check_rr_42",
            "_parity_check_rr_43",
            "_parity_check_rr_44",
            "_parity_check_rr_45",
            "_parity_check_rr_46",
            "_parity_check_rr_47",
            "_parity_check_rr_48",
            "_parity_check_rr_49",
            "_parity_check_rr_50",
            "_parity_check_rr_51",
            "_parity_check_rr_52",
            "_parity_check_rr_53",
            "_parity_check_rr_54",
            "_parity_check_rr_55",
            "_parity_check_rr_56",
            "_parity_check_rr_57",
            "_parity_check_rr_58",
            "_parity_check_cb_1",
            "_parity_check_cb_2",
            "_parity_check_cb_3",
            "_parity_check_cb_4",
            "_parity_check_cb_5",
            "_parity_check_cb_6",
            "_parity_check_cb_7",
            "_parity_check_cb_8",
            "_parity_check_cb_9",
            "_parity_check_cb_10",
            "_parity_check_cb_11",
            "_parity_check_cb_12",
            "_parity_check_cb_13",
            "_parity_check_cb_14",
            "_parity_check_cb_15",
            "_parity_check_cb_16",
            "_parity_check_cb_17",
            "_parity_check_cb_18",
            "_parity_check_cb_19",
            "_parity_check_cb_20",
            "_parity_check_cb_21",
            "_parity_check_cb_22",
            "_parity_check_cb_23",
            "_parity_check_cb_24",
            "_parity_check_cb_25",
            "_parity_check_cb_26",
            "_parity_check_cb_27",
            "_parity_check_cb_28",
            "_parity_check_cb_29",
            "_parity_check_cb_30",
            "_parity_check_cb_31",
            "_parity_check_cb_32",
            "_parity_check_cb_33",
            "_parity_check_cb_34",
            "_parity_check_cb_35",
            "_parity_check_cb_36",
            "_parity_check_cb_37",
            "_parity_check_cb_38",
            "_parity_check_cb_39",
            "_parity_check_cb_40",
            "_parity_check_cb_41",
            "_parity_check_cb_42",
            "_parity_check_cb_43",
            "_parity_check_cb_44",
            "_parity_check_cb_45",
            "_parity_check_cb_46",
            "_parity_check_cb_47",
            "_parity_check_cb_48",
            "_parity_check_cb_49",
            "_parity_check_cb_50",
            "_parity_check_cb_51",
            "_parity_check_cb_52",
            "_parity_check_cb_53",
            "_parity_check_cb_54",
            "_parity_check_cb_55",
            "_parity_check_cb_56",
            "_parity_check_cb_57",
            "_parity_check_cb_58"]
        return names

    def _constraints_function(self, data):
        """
        This is the function that effectively computes constraints on given data, must be implemented by the user.
        :param data: Data on which to compute the constraints function.
        :return:
        """
        area = self.experiment["AREA"]
        polygons_number = self.experiment["POLYGONS_NUMBER"]
        shape = self.experiment["SHAPE"]
        return self._area_and_all_parity_check_real_data(data, area, polygons_number, shape)

    def _area_and_all_parity_check_real_data(self, data_mb, area, polygons_number, shape):
        # preallocate an output array that will contain the computed
        # constraints values for the batch
        img_width = shape[0]
        img_height = shape[1]

        target_area = area * polygons_number
        nonzero = np_sum(data_mb[:, 1:-1, 1:-1], axis=(1, 2))
        norm = img_width * img_height - nonzero
        greater_area_inner = \
            np.minimum(1, np.maximum(0, nonzero - target_area) / norm)
        smaller_area_inner = \
            np.minimum(1, np.maximum(0, target_area - nonzero) / norm)
        sum_left = np_sum(
            data_mb[:, 1:(img_width - 1), 1:int(img_width / 2), 0], 2)
        check_left = np.mod(sum_left, 2)
        check_left = np.not_equal(
            data_mb[:, 1:(img_width - 1), 0, 0], check_left).astype(float)

        sum_top = np_sum(
            data_mb[:, 1:int(img_height / 2), 1:(img_height - 1), 0],
            1)
        check_top = np.mod(sum_top, 2)
        check_top = np.not_equal(
            data_mb[:, 0, 1:(img_height - 1), 0], check_top).astype(float)

        sum_right = np_sum(
            data_mb[:, 1:(img_width - 1), int(img_width / 2):(img_width - 1), 0], 2)
        check_right = np.mod(sum_right, 2)
        check_right = np.not_equal(
            data_mb[:, 1:(img_width - 1), img_width - 1, 0], check_right) \
            .astype(float)

        sum_bottom = np_sum(
            data_mb[:, int(img_height / 2):(img_height - 1), 1:(img_height - 1), 0],
            1)
        check_bottom = np.mod(sum_bottom, 2)
        check_bottom = np.not_equal(
            data_mb[:, (img_height - 1), 1:(img_height - 1), 0], check_bottom). \
            astype(float)
        mb_constraints_values = \
            np.concatenate([greater_area_inner, smaller_area_inner,
                            check_left, check_top, check_right, check_bottom],
                           axis=1)

        return mb_constraints_values


class AreaAndAllParityCheck32(ConstraintsComputable):
    def __init__(self, experiment, training_data, validation_data, test_data, graph_nodes):
        super().__init__(experiment, training_data, validation_data, test_data, graph_nodes)

    @staticmethod
    def constraints_names():
        """
        Returns a list of names, where ith name represent the ith constraint computed
        by the constraint function of this class. Make sure the length of this list
        is equal to the number of constraints computed for each data point.
        :return:
        """
        names = [
            "_greater_area_inner",
            "_smaller_area_inner",
            "_parity_check_rl_1",
            "_parity_check_rl_2",
            "_parity_check_rl_3",
            "_parity_check_rl_4",
            "_parity_check_rl_5",
            "_parity_check_rl_6",
            "_parity_check_rl_7",
            "_parity_check_rl_8",
            "_parity_check_rl_9",
            "_parity_check_rl_10",
            "_parity_check_rl_11",
            "_parity_check_rl_12",
            "_parity_check_rl_13",
            "_parity_check_rl_14",
            "_parity_check_rl_15",
            "_parity_check_rl_16",
            "_parity_check_rl_17",
            "_parity_check_rl_18",
            "_parity_check_rl_19",
            "_parity_check_rl_20",
            "_parity_check_rl_21",
            "_parity_check_rl_22",
            "_parity_check_rl_23",
            "_parity_check_rl_24",
            "_parity_check_rl_25",
            "_parity_check_rl_26",
            "_parity_check_rl_27",
            "_parity_check_rl_28",
            "_parity_check_rl_29",
            "_parity_check_rl_30",
            "_parity_check_ct_1",
            "_parity_check_ct_2",
            "_parity_check_ct_3",
            "_parity_check_ct_4",
            "_parity_check_ct_5",
            "_parity_check_ct_6",
            "_parity_check_ct_7",
            "_parity_check_ct_8",
            "_parity_check_ct_9",
            "_parity_check_ct_10",
            "_parity_check_ct_11",
            "_parity_check_ct_12",
            "_parity_check_ct_13",
            "_parity_check_ct_14",
            "_parity_check_ct_15",
            "_parity_check_ct_16",
            "_parity_check_ct_17",
            "_parity_check_ct_18",
            "_parity_check_ct_19",
            "_parity_check_ct_20",
            "_parity_check_ct_21",
            "_parity_check_ct_22",
            "_parity_check_ct_23",
            "_parity_check_ct_24",
            "_parity_check_ct_25",
            "_parity_check_ct_26",
            "_parity_check_ct_27",
            "_parity_check_ct_28",
            "_parity_check_ct_29",
            "_parity_check_ct_30",
            "_parity_check_rr_1",
            "_parity_check_rr_2",
            "_parity_check_rr_3",
            "_parity_check_rr_4",
            "_parity_check_rr_5",
            "_parity_check_rr_6",
            "_parity_check_rr_7",
            "_parity_check_rr_8",
            "_parity_check_rr_9",
            "_parity_check_rr_10",
            "_parity_check_rr_11",
            "_parity_check_rr_12",
            "_parity_check_rr_13",
            "_parity_check_rr_14",
            "_parity_check_rr_15",
            "_parity_check_rr_16",
            "_parity_check_rr_17",
            "_parity_check_rr_18",
            "_parity_check_rr_19",
            "_parity_check_rr_20",
            "_parity_check_rr_21",
            "_parity_check_rr_22",
            "_parity_check_rr_23",
            "_parity_check_rr_24",
            "_parity_check_rr_25",
            "_parity_check_rr_26",
            "_parity_check_rr_27",
            "_parity_check_rr_28",
            "_parity_check_rr_29",
            "_parity_check_rr_30",
            "_parity_check_cb_1",
            "_parity_check_cb_2",
            "_parity_check_cb_3",
            "_parity_check_cb_4",
            "_parity_check_cb_5",
            "_parity_check_cb_6",
            "_parity_check_cb_7",
            "_parity_check_cb_8",
            "_parity_check_cb_9",
            "_parity_check_cb_10",
            "_parity_check_cb_11",
            "_parity_check_cb_12",
            "_parity_check_cb_13",
            "_parity_check_cb_14",
            "_parity_check_cb_15",
            "_parity_check_cb_16",
            "_parity_check_cb_17",
            "_parity_check_cb_18",
            "_parity_check_cb_19",
            "_parity_check_cb_20",
            "_parity_check_cb_21",
            "_parity_check_cb_22",
            "_parity_check_cb_23",
            "_parity_check_cb_24",
            "_parity_check_cb_25",
            "_parity_check_cb_26",
            "_parity_check_cb_27",
            "_parity_check_cb_28",
            "_parity_check_cb_29",
            "_parity_check_cb_30"]
        return names

    def _constraints_function(self, data):
        """
        This is the function that effectively computes constraints on given data, must be implemented by the user.
        :param data: Data on which to compute the constraints function.
        :return:
        """
        area = self.experiment["AREA"]
        polygons_number = self.experiment["POLYGONS_NUMBER"]
        shape = self.experiment["SHAPE"]
        return self._area_and_all_parity_check_real_data(data, area, polygons_number, shape)

    def _area_and_all_parity_check_real_data(self, data_mb, area, polygons_number, shape):
        # preallocate an output array that will contain the computed
        # constraints values for the batch
        img_width = shape[0]
        img_height = shape[1]

        target_area = area * polygons_number
        nonzero = np_sum(data_mb[:, 1:-1, 1:-1], axis=(1, 2))
        norm = img_width * img_height - nonzero
        greater_area_inner = \
            np.minimum(1, np.maximum(0, nonzero - target_area) / norm)
        smaller_area_inner = \
            np.minimum(1, np.maximum(0, target_area - nonzero) / norm)
        sum_left = np_sum(
            data_mb[:, 1:(img_width - 1), 1:int(img_width / 2), 0], 2)
        check_left = np.mod(sum_left, 2)
        check_left = np.not_equal(
            data_mb[:, 1:(img_width - 1), 0, 0], check_left).astype(float)

        sum_top = np_sum(
            data_mb[:, 1:int(img_height / 2), 1:(img_height - 1), 0],
            1)
        check_top = np.mod(sum_top, 2)
        check_top = np.not_equal(
            data_mb[:, 0, 1:(img_height - 1), 0], check_top).astype(float)

        sum_right = np_sum(
            data_mb[:, 1:(img_width - 1), int(img_width / 2):(img_width - 1), 0], 2)
        check_right = np.mod(sum_right, 2)
        check_right = np.not_equal(
            data_mb[:, 1:(img_width - 1), img_width - 1, 0], check_right) \
            .astype(float)

        sum_bottom = np_sum(
            data_mb[:, int(img_height / 2):(img_height - 1), 1:(img_height - 1), 0],
            1)
        check_bottom = np.mod(sum_bottom, 2)
        check_bottom = np.not_equal(
            data_mb[:, (img_height - 1), 1:(img_height - 1), 0], check_bottom). \
            astype(float)
        mb_constraints_values = \
            np.concatenate([greater_area_inner, smaller_area_inner,
                            check_left, check_top, check_right, check_bottom],
                           axis=1)

        return mb_constraints_values


class AreaAndAllParityCheck20(ConstraintsComputable):
    def __init__(self, experiment, training_data, validation_data, test_data, graph_nodes):
        super().__init__(experiment, training_data, validation_data, test_data, graph_nodes)

    @staticmethod
    def constraints_names():
        """
        Returns a list of names, where ith name represent the ith constraint computed
        by the constraint function of this class. Make sure the length of this list
        is equal to the number of constraints computed for each data point.
        :return:
        """
        names = [
            "_greater_area_inner",
            "_smaller_area_inner",
            "_parity_check_rl_1",
            "_parity_check_rl_2",
            "_parity_check_rl_3",
            "_parity_check_rl_4",
            "_parity_check_rl_5",
            "_parity_check_rl_6",
            "_parity_check_rl_7",
            "_parity_check_rl_8",
            "_parity_check_rl_9",
            "_parity_check_rl_10",
            "_parity_check_rl_11",
            "_parity_check_rl_12",
            "_parity_check_rl_13",
            "_parity_check_rl_14",
            "_parity_check_rl_15",
            "_parity_check_rl_16",
            "_parity_check_rl_17",
            "_parity_check_rl_18",
            "_parity_check_ct_1",
            "_parity_check_ct_2",
            "_parity_check_ct_3",
            "_parity_check_ct_4",
            "_parity_check_ct_5",
            "_parity_check_ct_6",
            "_parity_check_ct_7",
            "_parity_check_ct_8",
            "_parity_check_ct_9",
            "_parity_check_ct_10",
            "_parity_check_ct_11",
            "_parity_check_ct_12",
            "_parity_check_ct_13",
            "_parity_check_ct_14",
            "_parity_check_ct_15",
            "_parity_check_ct_16",
            "_parity_check_ct_17",
            "_parity_check_ct_18",
            "_parity_check_rr_1",
            "_parity_check_rr_2",
            "_parity_check_rr_3",
            "_parity_check_rr_4",
            "_parity_check_rr_5",
            "_parity_check_rr_6",
            "_parity_check_rr_7",
            "_parity_check_rr_8",
            "_parity_check_rr_9",
            "_parity_check_rr_10",
            "_parity_check_rr_11",
            "_parity_check_rr_12",
            "_parity_check_rr_13",
            "_parity_check_rr_14",
            "_parity_check_rr_15",
            "_parity_check_rr_16",
            "_parity_check_rr_17",
            "_parity_check_rr_18",
            "_parity_check_cb_1",
            "_parity_check_cb_2",
            "_parity_check_cb_3",
            "_parity_check_cb_4",
            "_parity_check_cb_5",
            "_parity_check_cb_6",
            "_parity_check_cb_7",
            "_parity_check_cb_8",
            "_parity_check_cb_9",
            "_parity_check_cb_10",
            "_parity_check_cb_11",
            "_parity_check_cb_12",
            "_parity_check_cb_13",
            "_parity_check_cb_14",
            "_parity_check_cb_15",
            "_parity_check_cb_16",
            "_parity_check_cb_17",
            "_parity_check_cb_18"]
        return names

    def _constraints_function(self, data):
        """
        This is the function that effectively computes constraints on given data, must be implemented by the user.
        :param data: Data on which to compute the constraints function.
        :return:
        """
        area = self.experiment["AREA"]
        polygons_number = self.experiment["POLYGONS_NUMBER"]
        shape = self.experiment["SHAPE"]
        return self._area_and_all_parity_check_real_data(data, area, polygons_number, shape)

    def _area_and_all_parity_check_real_data(self, data_mb, area, polygons_number, shape):
        # preallocate an output array that will contain the computed
        # constraints values for the batch
        img_width = shape[0]
        img_height = shape[1]

        target_area = area * polygons_number
        nonzero = np_sum(data_mb[:, 1:-1, 1:-1], axis=(1, 2))
        norm = img_width * img_height - nonzero
        greater_area_inner = \
            np.minimum(1, np.maximum(0, nonzero - target_area) / norm)
        smaller_area_inner = \
            np.minimum(1, np.maximum(0, target_area - nonzero) / norm)
        sum_left = np_sum(
            data_mb[:, 1:(img_width - 1), 1:int(img_width / 2), 0], 2)
        check_left = np.mod(sum_left, 2)
        check_left = np.not_equal(
            data_mb[:, 1:(img_width - 1), 0, 0], check_left).astype(float)

        sum_top = np_sum(
            data_mb[:, 1:int(img_height / 2), 1:(img_height - 1), 0],
            1)
        check_top = np.mod(sum_top, 2)
        check_top = np.not_equal(
            data_mb[:, 0, 1:(img_height - 1), 0], check_top).astype(float)

        sum_right = np_sum(
            data_mb[:, 1:(img_width - 1), int(img_width / 2):(img_width - 1), 0], 2)
        check_right = np.mod(sum_right, 2)
        check_right = np.not_equal(
            data_mb[:, 1:(img_width - 1), img_width - 1, 0], check_right) \
            .astype(float)

        sum_bottom = np_sum(
            data_mb[:, int(img_height / 2):(img_height - 1), 1:(img_height - 1), 0],
            1)
        check_bottom = np.mod(sum_bottom, 2)
        check_bottom = np.not_equal(
            data_mb[:, (img_height - 1), 1:(img_height - 1), 0], check_bottom). \
            astype(float)
        mb_constraints_values = \
            np.concatenate([greater_area_inner, smaller_area_inner,
                            check_left, check_top, check_right, check_bottom],
                           axis=1)
        return mb_constraints_values


class AllParityCheckMulti16(ConstraintsComputable):
    def __init__(self, experiment, training_data, validation_data, test_data, graph_nodes):
        super().__init__(experiment, training_data, validation_data, test_data, graph_nodes)

    @staticmethod
    def constraints_names():
        """
        Returns a list of names, where ith name represent the ith constraint computed
        by the constraint function of this class. Make sure the length of this list
        is equal to the number of constraints computed for each data point.
        :return:
        """
        names = [
            "_parity_check_rl_1",
            "_parity_check_rl_2",
            "_parity_check_rl_3",
            "_parity_check_rl_4",
            "_parity_check_rl_5",
            "_parity_check_rl_6",
            "_parity_check_rl_7",
            "_parity_check_rl_8",
            "_parity_check_rl_9",
            "_parity_check_rl_10",
            "_parity_check_rl_11",
            "_parity_check_rl_12",
            "_parity_check_rl_13",
            "_parity_check_rl_14",
            "_parity_check_ct_1",
            "_parity_check_ct_2",
            "_parity_check_ct_3",
            "_parity_check_ct_4",
            "_parity_check_ct_5",
            "_parity_check_ct_6",
            "_parity_check_ct_7",
            "_parity_check_ct_8",
            "_parity_check_ct_9",
            "_parity_check_ct_10",
            "_parity_check_ct_11",
            "_parity_check_ct_12",
            "_parity_check_ct_13",
            "_parity_check_ct_14",
            "_parity_check_rr_1",
            "_parity_check_rr_2",
            "_parity_check_rr_3",
            "_parity_check_rr_4",
            "_parity_check_rr_5",
            "_parity_check_rr_6",
            "_parity_check_rr_7",
            "_parity_check_rr_8",
            "_parity_check_rr_9",
            "_parity_check_rr_10",
            "_parity_check_rr_11",
            "_parity_check_rr_12",
            "_parity_check_rr_13",
            "_parity_check_rr_14",
            "_parity_check_cb_1",
            "_parity_check_cb_2",
            "_parity_check_cb_3",
            "_parity_check_cb_4",
            "_parity_check_cb_5",
            "_parity_check_cb_6",
            "_parity_check_cb_7",
            "_parity_check_cb_8",
            "_parity_check_cb_9",
            "_parity_check_cb_10",
            "_parity_check_cb_11",
            "_parity_check_cb_12",
            "_parity_check_cb_13",
            "_parity_check_cb_14"]
        return names

    def _constraints_function(self, data):
        """
        This is the function that effectively computes constraints on given data, must be implemented by the user.
        :param data: Data on which to compute the constraints function.
        :return:
        """
        area = self.experiment["AREA"]
        polygons_number = self.experiment["POLYGONS_NUMBER"]
        shape = self.experiment["SHAPE"]
        color_map = self.experiment["COLOR_MAP"]
        return self._all_parity_check_real_data_multi(data, area, polygons_number, shape, color_map)

    def _all_parity_check_real_data_multi(self, data_mb, area, polygons_number, shape, color_map):
        img_width = shape[0]
        img_height = shape[1]
        white_index = color_map[(255, 255, 255)]
        black_index = color_map[(0, 0, 0)]
        sum_left = np_sum(
            data_mb[:, 1:(img_width - 1), 1:int(img_width / 2), black_index] == 0,
            2)
        check_left = np.mod(sum_left, 2)
        check_left = np.not_equal(
            data_mb[:, 1:(img_width - 1), 0, white_index], check_left).astype(float)

        sum_top = np_sum(
            data_mb[:, 1:int(img_height / 2), 1:(img_height - 1), black_index] == 0,
            1)
        check_top = np.mod(sum_top, 2)
        check_top = np.not_equal(
            data_mb[:, 0, 1:(img_height - 1), white_index], check_top).astype(float)

        sum_right = np_sum(
            data_mb[:, 1:(img_width - 1), int(img_width / 2):(img_width - 1),
            black_index] == 0, 2)
        check_right = np.mod(sum_right, 2)
        check_right = np.not_equal(
            data_mb[:, 1:(img_width - 1), img_width - 1, white_index], check_right) \
            .astype(float)

        sum_bottom = np_sum(
            data_mb[:, int(img_height / 2):(img_height - 1), 1:(img_height - 1),
            black_index] == 0, 1)
        check_bottom = np.mod(sum_bottom, 2)
        check_bottom = np.not_equal(
            data_mb[:, (img_height - 1), 1:(img_height - 1), white_index],
            check_bottom).astype(float)
        mb_constraints_values = \
            np.concatenate([check_left, check_top, check_right, check_bottom],
                           axis=1)

        return mb_constraints_values


class AllParityCheckMulti20(ConstraintsComputable):
    def __init__(self, experiment, training_data, validation_data, test_data, graph_nodes):
        super().__init__(experiment, training_data, validation_data, test_data, graph_nodes)

    @staticmethod
    def constraints_names():
        """
        Returns a list of names, where ith name represent the ith constraint computed
        by the constraint function of this class. Make sure the length of this list
        is equal to the number of constraints computed for each data point.
        :return:
        """
        names = [
            "_parity_check_rl_1",
            "_parity_check_rl_2",
            "_parity_check_rl_3",
            "_parity_check_rl_4",
            "_parity_check_rl_5",
            "_parity_check_rl_6",
            "_parity_check_rl_7",
            "_parity_check_rl_8",
            "_parity_check_rl_9",
            "_parity_check_rl_10",
            "_parity_check_rl_11",
            "_parity_check_rl_12",
            "_parity_check_rl_13",
            "_parity_check_rl_14",
            "_parity_check_rl_15",
            "_parity_check_rl_16",
            "_parity_check_rl_17",
            "_parity_check_rl_18",
            "_parity_check_ct_1",
            "_parity_check_ct_2",
            "_parity_check_ct_3",
            "_parity_check_ct_4",
            "_parity_check_ct_5",
            "_parity_check_ct_6",
            "_parity_check_ct_7",
            "_parity_check_ct_8",
            "_parity_check_ct_9",
            "_parity_check_ct_10",
            "_parity_check_ct_11",
            "_parity_check_ct_12",
            "_parity_check_ct_13",
            "_parity_check_ct_14",
            "_parity_check_ct_15",
            "_parity_check_ct_16",
            "_parity_check_ct_17",
            "_parity_check_ct_18",
            "_parity_check_rr_1",
            "_parity_check_rr_2",
            "_parity_check_rr_3",
            "_parity_check_rr_4",
            "_parity_check_rr_5",
            "_parity_check_rr_6",
            "_parity_check_rr_7",
            "_parity_check_rr_8",
            "_parity_check_rr_9",
            "_parity_check_rr_10",
            "_parity_check_rr_11",
            "_parity_check_rr_12",
            "_parity_check_rr_13",
            "_parity_check_rr_14",
            "_parity_check_rr_15",
            "_parity_check_rr_16",
            "_parity_check_rr_17",
            "_parity_check_rr_18",
            "_parity_check_cb_1",
            "_parity_check_cb_2",
            "_parity_check_cb_3",
            "_parity_check_cb_4",
            "_parity_check_cb_5",
            "_parity_check_cb_6",
            "_parity_check_cb_7",
            "_parity_check_cb_8",
            "_parity_check_cb_9",
            "_parity_check_cb_10",
            "_parity_check_cb_11",
            "_parity_check_cb_12",
            "_parity_check_cb_13",
            "_parity_check_cb_14",
            "_parity_check_cb_15",
            "_parity_check_cb_16",
            "_parity_check_cb_17",
            "_parity_check_cb_18"]
        return names

    def _constraints_function(self, data):
        """
        This is the function that effectively computes constraints on given data, must be implemented by the user.
        :param data: Data on which to compute the constraints function.
        :return:
        """
        area = self.experiment["AREA"]
        polygons_number = self.experiment["POLYGONS_NUMBER"]
        shape = self.experiment["SHAPE"]
        color_map = self.experiment["COLOR_MAP"]
        return self._all_parity_check_real_data_multi(data, area, polygons_number, shape, color_map)

    def _all_parity_check_real_data_multi(self, data_mb, area, polygons_number, shape, color_map):
        img_width = shape[0]
        img_height = shape[1]
        white_index = color_map[(255, 255, 255)]
        black_index = color_map[(0, 0, 0)]
        sum_left = np_sum(
            data_mb[:, 1:(img_width - 1), 1:int(img_width / 2), black_index] == 0,
            2)
        check_left = np.mod(sum_left, 2)
        check_left = np.not_equal(
            data_mb[:, 1:(img_width - 1), 0, white_index], check_left).astype(float)

        sum_top = np_sum(
            data_mb[:, 1:int(img_height / 2), 1:(img_height - 1), black_index] == 0,
            1)
        check_top = np.mod(sum_top, 2)
        check_top = np.not_equal(
            data_mb[:, 0, 1:(img_height - 1), white_index], check_top).astype(float)

        sum_right = np_sum(
            data_mb[:, 1:(img_width - 1), int(img_width / 2):(img_width - 1),
            black_index] == 0, 2)
        check_right = np.mod(sum_right, 2)
        check_right = np.not_equal(
            data_mb[:, 1:(img_width - 1), img_width - 1, white_index], check_right) \
            .astype(float)

        sum_bottom = np_sum(
            data_mb[:, int(img_height / 2):(img_height - 1), 1:(img_height - 1),
            black_index] == 0, 1)
        check_bottom = np.mod(sum_bottom, 2)
        check_bottom = np.not_equal(
            data_mb[:, (img_height - 1), 1:(img_height - 1), white_index],
            check_bottom).astype(float)
        mb_constraints_values = \
            np.concatenate([check_left, check_top, check_right, check_bottom],
                           axis=1)

        return mb_constraints_values


class AreaAndAllParityCheckMultiBW20(ConstraintsComputable):
    def __init__(self, experiment, training_data, validation_data, test_data, graph_nodes):
        super().__init__(experiment, training_data, validation_data, test_data, graph_nodes)

    @staticmethod
    def constraints_names():
        """
        Returns a list of names, where ith name represent the ith constraint computed
        by the constraint function of this class. Make sure the length of this list
        is equal to the number of constraints computed for each data point.
        :return:
        """
        names = [
            "_greater_area_inner",
            "_smaller_area_inner",
            "_parity_check_rl_1",
            "_parity_check_rl_2",
            "_parity_check_rl_3",
            "_parity_check_rl_4",
            "_parity_check_rl_5",
            "_parity_check_rl_6",
            "_parity_check_rl_7",
            "_parity_check_rl_8",
            "_parity_check_rl_9",
            "_parity_check_rl_10",
            "_parity_check_rl_11",
            "_parity_check_rl_12",
            "_parity_check_rl_13",
            "_parity_check_rl_14",
            "_parity_check_rl_15",
            "_parity_check_rl_16",
            "_parity_check_rl_17",
            "_parity_check_rl_18",
            "_parity_check_ct_1",
            "_parity_check_ct_2",
            "_parity_check_ct_3",
            "_parity_check_ct_4",
            "_parity_check_ct_5",
            "_parity_check_ct_6",
            "_parity_check_ct_7",
            "_parity_check_ct_8",
            "_parity_check_ct_9",
            "_parity_check_ct_10",
            "_parity_check_ct_11",
            "_parity_check_ct_12",
            "_parity_check_ct_13",
            "_parity_check_ct_14",
            "_parity_check_ct_15",
            "_parity_check_ct_16",
            "_parity_check_ct_17",
            "_parity_check_ct_18",
            "_parity_check_rr_1",
            "_parity_check_rr_2",
            "_parity_check_rr_3",
            "_parity_check_rr_4",
            "_parity_check_rr_5",
            "_parity_check_rr_6",
            "_parity_check_rr_7",
            "_parity_check_rr_8",
            "_parity_check_rr_9",
            "_parity_check_rr_10",
            "_parity_check_rr_11",
            "_parity_check_rr_12",
            "_parity_check_rr_13",
            "_parity_check_rr_14",
            "_parity_check_rr_15",
            "_parity_check_rr_16",
            "_parity_check_rr_17",
            "_parity_check_rr_18",
            "_parity_check_cb_1",
            "_parity_check_cb_2",
            "_parity_check_cb_3",
            "_parity_check_cb_4",
            "_parity_check_cb_5",
            "_parity_check_cb_6",
            "_parity_check_cb_7",
            "_parity_check_cb_8",
            "_parity_check_cb_9",
            "_parity_check_cb_10",
            "_parity_check_cb_11",
            "_parity_check_cb_12",
            "_parity_check_cb_13",
            "_parity_check_cb_14",
            "_parity_check_cb_15",
            "_parity_check_cb_16",
            "_parity_check_cb_17",
            "_parity_check_cb_18"]
        return names

    def _constraints_function(self, data):
        """
        This is the function that effectively computes constraints on given data, must be implemented by the user.
        :param data: Data on which to compute the constraints function.
        :return:
        """
        area = self.experiment["AREA"]
        polygons_number = self.experiment["POLYGONS_NUMBER"]
        shape = self.experiment["SHAPE"]
        color_map = self.experiment["COLOR_MAP"]
        return self._area_and_all_parity_check_real_data_multi_BW(data, area, polygons_number, shape, color_map)

    def _area_and_all_parity_check_real_data_multi_BW(self, data_mb, area, polygons_number, shape, color_map):
        # we have (n_samples x bs) generated items; for each of them we have to
        # compute the values of each constraint.
        # R_mb has shape [n_samples, bs] + image_shape
        # e.g. [n_samples, bs, 20, 20, 1]
        # so we need to iterate over the first two dimensions
        img_width = shape[0]
        img_height = shape[1]
        white_index = color_map[(255, 255, 255)]
        black_index = color_map[(0, 0, 0)]
        target_area = area * polygons_number
        nonblack = np_sum(data_mb[:, 1:-1, 1:-1, black_index] == 0, axis=(1, 2))
        nonblack = nonblack.reshape(nonblack.shape + (1,))

        norm = img_width * img_height - target_area
        greater_area_inner = np.minimum(1, np.maximum(0,
                                                      nonblack - target_area) / norm)
        smaller_area_inner = np.minimum(1, np.maximum(0,
                                                      target_area - nonblack) / norm)
        sum_left = np_sum(
            data_mb[:, 1:(img_width - 1), 1:int(img_width / 2), black_index] == 0,
            2)
        check_left = np.mod(sum_left, 2)
        check_left = np.not_equal(
            data_mb[:, 1:(img_width - 1), 0, white_index], check_left).astype(float)

        sum_top = np_sum(
            data_mb[:, 1:int(img_height / 2), 1:(img_height - 1), black_index] == 0,
            1)
        check_top = np.mod(sum_top, 2)
        check_top = np.not_equal(
            data_mb[:, 0, 1:(img_height - 1), white_index], check_top).astype(float)

        sum_right = np_sum(
            data_mb[:, 1:(img_width - 1), int(img_width / 2):(img_width - 1),
            black_index] == 0, 2)
        check_right = np.mod(sum_right, 2)
        check_right = np.not_equal(
            data_mb[:, 1:(img_width - 1), img_width - 1, white_index], check_right) \
            .astype(float)

        sum_bottom = np_sum(
            data_mb[:, int(img_height / 2):(img_height - 1), 1:(img_height - 1),
            black_index] == 0, 1)
        check_bottom = np.mod(sum_bottom, 2)
        check_bottom = np.not_equal(
            data_mb[:, (img_height - 1), 1:(img_height - 1), white_index],
            check_bottom).astype(float)
        mb_constraints_values = \
            np.concatenate([greater_area_inner, smaller_area_inner, check_left,
                            check_top, check_right, check_bottom], axis=1)

        return mb_constraints_values


class AreaAndConvexity(ConstraintsComputable):
    def __init__(self, experiment, training_data, validation_data, test_data, graph_nodes):
        super().__init__(experiment, training_data, validation_data, test_data, graph_nodes)

    @staticmethod
    def constraints_names():
        """
        Returns a list of names, where ith name represent the ith constraint computed
        by the constraint function of this class. Make sure the length of this list
        is equal to the number of constraints computed for each data point.
        :return:
        """
        names = [
            "_greater_area_inner",
            "_smaller_area_inner",
            "_convex"]
        return names

    def _constraints_function(self, data):
        """
        This is the function that effectively computes constraints on given data, must be implemented by the user.
        :param data: Data on which to compute the constraints function.
        :return:
        """
        area = self.experiment["AREA"]
        polygons_number = self.experiment["POLYGONS_NUMBER"]
        shape = self.experiment["SHAPE"]
        return self._area_and_convexity_single_batch(data, area, polygons_number, shape)

    def _area_and_convexity_single_batch(self, data, area, polygons_number, shape):
        # preallocate an output array that will contain the 3 computed constraints values for the batch

        mb_constraints_values = np_empty(shape=(data.shape[0], 3), dtype=np_float32)

        for i in range(data.shape[0]):
            sample = data[i]

            target_area = area * polygons_number
            nonzero = np_sum(sample)
            norm = shape[0] * shape[1] - target_area
            greater_area_inner = min(1, max(0, nonzero - target_area) / norm)
            smaller_area_inner = min(1, max(0, target_area - nonzero) / norm)

            mb_constraints_values[i] = [greater_area_inner, smaller_area_inner, _convex(sample, shape[0], shape[1])]
        return mb_constraints_values


class AreaMulti(ConstraintsComputable):
    def __init__(self, experiment, training_data, validation_data, test_data, graph_nodes):
        super().__init__(experiment, training_data, validation_data, test_data, graph_nodes)

    @staticmethod
    def constraints_names():
        """
        Returns a list of names, where ith name represent the ith constraint computed
        by the constraint function of this class. Make sure the length of this list
        is equal to the number of constraints computed for each data point.
        :return:
        """
        names = [
            "_greater_area_inner",
            "_smaller_area_inner"]
        return names

    def _constraints_function(self, data):
        """
        This is the function that effectively computes constraints on given data, must be implemented by the user.
        :param data: Data on which to compute the constraints function.
        :return:
        """
        area = self.experiment["AREA"]
        polygons_number = self.experiment["POLYGONS_NUMBER"]
        shape = self.experiment["SHAPE"]
        color_map = self.experiment["COLOR_MAP"]
        return self._area_real_data_multi(data, area, polygons_number, shape, color_map)

    # @jit(nopython=True, parallel=True)
    def _area_real_data_multi(self, data, area, polygons_number, shape, color_map):
        # index of white and black colors
        white_index = color_map[(255, 255, 255)]
        black_index = color_map[(0, 0, 0)]

        # expected number of total white pixels
        target_area = area * polygons_number

        nonblack = np_sum(data[:, 1:-1, 1:-1, black_index] == 0, axis=(1, 2))
        nonblack = nonblack.reshape(nonblack.shape + (1,))

        norm = shape[0] * shape[1] - target_area
        greater_area_inner = np.minimum(1, np.maximum(0, nonblack - target_area) / norm)
        smaller_area_inner = np.minimum(1, np.maximum(0, target_area - nonblack) / norm)

        return np.concatenate([greater_area_inner, smaller_area_inner], axis=1)


class FloorPlanningConstraints(ConstraintsComputable):
    def __init__(self, experiment, training_data, validation_data, test_data, graph_nodes):
        super().__init__(experiment, training_data, validation_data, test_data, graph_nodes)

    @staticmethod
    def constraints_names():
        """
        Returns a list of names, where ith name represent the ith constraint computed
        by the constraint function of this class. Make sure the length of this list
        is equal to the number of constraints computed for each data point.
        :return: the list of constraints names
        """
        names = [
            "_greater_common_room_area_inner",
            "_smaller_common_room_area_inner",
            "_greater_living_area_inner",
            "_smaller_living_area_inner",
            "_greater_resting_area_inner",
            "_smaller_resting_area_inner",
            "_greater_kitchen_area_inner",
            "_smaller_kitchen_area_inner",
            "_greater_living_room_area_inner",
            "_smaller_living_room_area_inner",
            "_greater_door_area_side",
            "_smaller_door_area_side",
            "_greater_door_area_angle",
            "_smaller_door_area_angle",
            "_greater_door_area_total",
            "_smaller_door_area_total",
            "_greater_door_area_inner",
            "_smaller_door_area_inner"
        ]
        return names

    def _constraints_function(self, data):
        """
        This is the function that effectively computes constraints on given data, must be implemented by the user.
        :param data: Data on which to compute the constraints function.
        :return: the computed constraints
        """
        apt_area = self.experiment["APT_AREA"]
        shape = self.experiment["SHAPE"]
        color_map = self.experiment["COLOR_MAP"]
        return self._floor_planning_constraints_real(data, shape, apt_area, color_map)

    # @jit(nopython=True, parallel=True)
    def _floor_planning_constraints_real(self, data, shape, apt_area, color_map):
        img_width = shape[0]
        img_height = shape[1]

        # common area
        common_room_area = apt_area[0] - 1
        common_room_index = color_map[(255, 255, 255)]
        common_area_pixels = np_sum(data[:, :, :, common_room_index] == 1, axis=(1, 2))
        common_area_pixels = common_area_pixels.reshape(common_area_pixels.shape + (1,))

        norm = img_width * img_height - common_room_area
        greater_common_room_area_inner = np.minimum(1, np.maximum(0, common_area_pixels - common_room_area) / norm)
        smaller_common_room_area_inner = np.minimum(1, np.maximum(0, common_room_area - common_area_pixels) / norm)

        # living area
        living_area_area = apt_area[-2]
        living_area_index = color_map[(255, 165, 0)]
        living_area_pixels = np_sum(data[:, :, :, living_area_index] == 1, axis=(1, 2))
        living_area_pixels = living_area_pixels.reshape(living_area_pixels.shape + (1,))

        norm = img_width * img_height - living_area_area
        greater_living_area_inner = np.minimum(1, np.maximum(0, living_area_pixels - living_area_area) / norm)
        smaller_living_area_inner = np.minimum(1, np.maximum(0, living_area_area - living_area_pixels) / norm)

        # resting area
        resting_area_area = apt_area[-1]
        resting_area_index = color_map[(255, 0, 255)]
        resting_area_pixels = np_sum(data[:, :, :, resting_area_index] == 1, axis=(1, 2))
        resting_area_pixels = resting_area_pixels.reshape(resting_area_pixels.shape + (1,))

        norm = img_width * img_height - resting_area_area
        greater_resting_area_inner = np.minimum(1, np.maximum(0, resting_area_pixels - resting_area_area) / norm)
        smaller_resting_area_inner = np.minimum(1, np.maximum(0, resting_area_area - resting_area_pixels) / norm)

        # kitchen
        kitchen_area = apt_area[1]
        kitchen_index = color_map[(255, 255, 0)]
        kitchen_pixels = np_sum(data[:, :, :, kitchen_index] == 1, axis=(1, 2))
        kitchen_pixels = kitchen_pixels.reshape(kitchen_pixels.shape + (1,))

        norm = img_width * img_height - kitchen_area
        greater_kitchen_area_inner = np.minimum(1, np.maximum(0, kitchen_pixels - kitchen_area) / norm)
        smaller_kitchen_area_inner = np.minimum(1, np.maximum(0, kitchen_area - kitchen_pixels) / norm)

        # living room
        living_room_area = apt_area[5]
        living_room_index = color_map[(0, 0, 255)]
        living_room_pixels = np_sum(data[:, :, :, living_room_index] == 1, axis=(1, 2))
        living_room_pixels = living_room_pixels.reshape(living_room_pixels.shape + (1,))

        norm = img_width * img_height - living_room_area
        greater_living_room_area_inner = np.minimum(1, np.maximum(0, living_room_pixels - living_room_area) / norm)
        smaller_living_room_area_inner = np.minimum(1, np.maximum(0, living_room_area - living_room_pixels) / norm)

        # doors
        door_index = color_map[(128, 0, 0)]

        door_area_angle = 0
        door_area_total = 1
        door_area_inner = 0
        door_area_side = 1

        # door area side part
        door_pixel_left = np_sum(data[:, 0, 1:img_height - 1, door_index], axis=1)
        door_pixel_left = door_pixel_left.reshape(door_pixel_left.shape + (1,))

        door_pixel_up = np_sum(data[:, 1:img_width - 1, 0, door_index], axis=1)
        door_pixel_up = door_pixel_up.reshape(door_pixel_up.shape + (1,))

        door_pixel_right = np_sum(data[:, img_width - 1, 1:img_height - 1, door_index], axis=1)
        door_pixel_right = door_pixel_right.reshape(door_pixel_right.shape + (1,))

        door_pixel_down = np_sum(data[:, 1:img_width - 1, img_height - 1, door_index], axis=1)
        door_pixel_down = door_pixel_down.reshape(door_pixel_down.shape + (1,))

        door_pixels_side = door_pixel_down + door_pixel_left + door_pixel_right + door_pixel_up

        norm = img_width * img_height - door_area_side
        greater_door_area_side = np.minimum(1, np.maximum(0, door_pixels_side - door_area_side) / norm)
        smaller_door_area_side = np.minimum(1, np.maximum(0, door_area_side - door_pixels_side) / norm)

        # door are angle part
        door_on_angle = data[:, 0, 0, door_index] + \
                        data[:, 0, img_height - 1, door_index] + \
                        data[:, img_width - 1, 0, door_index] + \
                        data[:, img_width - 1, img_height - 1, door_index]
        door_on_angle = door_on_angle.reshape(door_on_angle.shape + (1,))

        norm = img_width * img_height - door_area_angle

        greater_door_area_angle = np.minimum(1, np.maximum(0, door_on_angle - door_area_angle) / norm)
        smaller_door_area_angle = np.minimum(1, np.maximum(0, door_area_angle - door_on_angle) / norm)

        # door area total part
        door_total_area_pixels = np_sum(data[:, :, :, door_index] == 1, axis=(1, 2))
        door_total_area_pixels = door_total_area_pixels.reshape(door_total_area_pixels.shape + (1,))

        norm = img_width * img_height - door_area_total

        greater_door_area_total = np.minimum(1, np.maximum(0, door_total_area_pixels - door_area_total) / norm)
        smaller_door_area_total = np.minimum(1, np.maximum(0, door_area_total - door_total_area_pixels) / norm)

        # door area inner part
        door_area_inner_pixels = np_sum(data[:, 1:-1, 1:-1, door_index] == 1, axis=(1, 2))
        door_area_inner_pixels = door_area_inner_pixels.reshape(door_area_inner_pixels.shape + (1,))

        norm = img_width * img_height - door_area_inner

        greater_door_area_inner = np.minimum(1, np.maximum(0, door_area_inner_pixels - door_area_inner) / norm)
        smaller_door_area_inner = np.minimum(1, np.maximum(0, door_area_inner - door_area_inner_pixels) / norm)

        return np.concatenate([greater_common_room_area_inner,
                               smaller_common_room_area_inner,
                               greater_living_area_inner,
                               smaller_living_area_inner,
                               greater_resting_area_inner,
                               smaller_resting_area_inner,
                               greater_kitchen_area_inner,
                               smaller_kitchen_area_inner,
                               greater_living_room_area_inner,
                               smaller_living_room_area_inner,
                               greater_door_area_side,
                               smaller_door_area_side,
                               greater_door_area_angle,
                               smaller_door_area_angle,
                               greater_door_area_total,
                               smaller_door_area_total,
                               greater_door_area_inner,
                               smaller_door_area_inner], axis=1)


def _convex(sample, img_width, img_height):
    # sample shape [20, 20, 1]
    sample = sample.reshape(sample.shape[:-1])

    visited_pixel = {(0, 0)}  # trick to let Numba infer set type
    visited_pixel.clear()  # remove placeholder element
    error = 0
    for w in range(img_width):
        for h in range(img_height):
            c = (w, h)
            if c in visited_pixel or sample[c] != 1:
                continue
            scc = {c}  # trick to let Numba infer set type
            _bfs(sample, visited_pixel, c, scc, img_width, img_height)

            # compute SCC error

            # partition by row
            for i in range(img_height):
                error += _axis_error([p for p in scc if p[1] == i], 0)

            # partition by col
            for i in range(img_width):
                error += _axis_error([p for p in scc if p[0] == i], 1)

            # partition by diag_R
            for i in range(img_width + img_height):
                error += _axis_error([p for p in scc if p[0] + p[1] == i], 0)

            # partition by diag_L
            for i in range(-img_height, img_width):
                error += _axis_error([p for p in scc if p[1] - p[0] == i], 0)
    return min(1, error / (2 * (img_width * img_height)))


# @jit(nopython=True, parallel=True)
def _min_max(items):
    assert len(items) > 0
    return min(items), max(items)


# @jit(nopython=True, parallel=True)
def _axis_error(iterable, axis):
    items = [i[axis] for i in iterable]
    if len(items) == 0:
        return 0
    min_item, max_item = _min_max(items)
    return max_item - min_item - len(items) + 1


# @jit(nopython=True, parallel=True)
def _bfs(sample, visited_pixel, c, scc, img_width, img_height):
    scc.add(c)
    visited_pixel.add(c)
    x, y = c

    up_value = c[0] - 1, c[1]
    if x > 0 and sample[up_value] == 1 and up_value not in visited_pixel:
        _bfs(sample, visited_pixel, up_value, scc, img_width, img_height)

    up_l_value = c[0] - 1, c[1] - 1
    if x > 0 and y > 0 and sample[up_l_value] == 1 and up_l_value not in visited_pixel:
        _bfs(sample, visited_pixel, up_l_value, scc, img_width, img_height)

    up_r_value = c[0] - 1, c[1] + 1
    if x > 0 and y < img_width - 2 and sample[up_r_value] == 1 and up_r_value not in visited_pixel:
        _bfs(sample, visited_pixel, up_r_value, scc, img_width, img_height)

    l_value = c[0], c[1] - 1
    if y > 0 and sample[l_value] == 1 and l_value not in visited_pixel:
        _bfs(sample, visited_pixel, l_value, scc, img_width, img_height)

    r_value = c[0], c[1] + 1
    if y < img_width - 2 and sample[r_value] == 1 and r_value not in visited_pixel:
        _bfs(sample, visited_pixel, r_value, scc, img_width, img_height)

    down_value = c[0] + 1, c[1]
    if x < img_height - 2 and sample[down_value] == 1 and down_value not in visited_pixel:
        _bfs(sample, visited_pixel, down_value, scc, img_width, img_height)

    down_l_value = c[0] + 1, c[1] - 1
    if x < img_height - 2 and y > 0 and sample[down_l_value] == 1 and down_l_value not in visited_pixel:
        _bfs(sample, visited_pixel, down_l_value, scc, img_width, img_height)

    down_r_value = c[0] + 1, c[1] + 1
    if x < img_height - 2 and y < img_width - 2 and sample[down_r_value] == 1 and down_r_value not in visited_pixel:
        _bfs(sample, visited_pixel, down_r_value, scc, img_width, img_height)
