import tensorflow as tf
from base_layers import Loss
from utils.utils_tf import pad_left
import numpy as np

class _LevelLoss(Loss):

    def _pre_processing(self, **graph_nodes):
        self._requires("D_fake", **graph_nodes)

# FakeLoss is used only for testing purposes
class FakeLoss(_LevelLoss):

    def _forward(self, **graph_nodes):
        res = dict()
        res["G_loss"] = tf.constant(0, dtype=tf.float32)
        return res, dict(), dict()


########################################################################################################################
############################################## GENERATOR LOSSES ########################################################
########################################################################################################################

"""
Main loss of the generator, taken and adapted from the https://github.com/TheHedgeify/DagstuhlGAN/tree/master/pytorch project.
"""

class DCGANLossGenerator(_LevelLoss):

    def _forward(self, **graph_nodes):
        D_fake = graph_nodes["D_fake"]

        G_loss = -tf.reduce_mean(D_fake)
        res = dict()
        res["G_loss"] = G_loss

        res_log = dict()
        res_log["G_main_loss"] = G_loss

        return res, res_log, dict()


class DCGANLossGeneratorSoftplus(_LevelLoss):

    def _forward(self, **graph_nodes):
        D_fake = graph_nodes["D_fake"]

        # Applying softplus both to the generator and discriminator loss, seems to improve performances
        G_loss = tf.reduce_mean(tf.nn.softplus(D_fake))
        res = dict()
        res["G_loss"] = G_loss

        res_log = dict()
        res_log["G_main_loss"] = G_loss

        return res, res_log, dict()


class DCGANLossGeneratorBinaryCrossentropy(_LevelLoss):

    def __init__(self, experiment, *args, **kwargs):
        super().__init__(experiment, *args, **kwargs)

        self.binary_cross_entropy_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def _forward(self, **graph_nodes):
        D_fake = graph_nodes["D_fake"]

        G_loss = self.binary_cross_entropy_loss(tf.zeros_like(D_fake), D_fake)
        res = dict()
        res["G_loss"] = G_loss

        res_log = dict()
        res_log["G_main_loss"] = G_loss

        return res, res_log, dict()

########################################################################################################################
############################################## DISCRIMINATOR LOSSES ####################################################
########################################################################################################################

"""
Main loss of the discriminator, taken and adapted from the https://github.com/TheHedgeify/DagstuhlGAN/tree/master/pytorch project.
"""


class DCGANLossDiscriminator(_LevelLoss):

    def _pre_processing(self, **graph_nodes):
        super()._pre_processing(**graph_nodes)
        self._requires("D_real", **graph_nodes)

    def _forward(self, **graph_nodes):
        D_real = graph_nodes["D_real"]
        D_fake = graph_nodes["D_fake"]

        D_loss = tf.reduce_mean(D_fake) - tf.reduce_mean(D_real)
        res = dict()
        res["D_loss"] = D_loss

        res_log = dict()
        res_log["D_main_loss"] = D_loss
        res_log["D_main_loss_fake"] = tf.reduce_mean(D_fake)
        res_log["D_main_loss_real"] = tf.reduce_mean(D_real)

        return res, res_log, dict()


class DCGANLossDiscriminatorSoftplus(DCGANLossDiscriminator):

    def _forward(self, **graph_nodes):
        D_real = graph_nodes["D_real"]
        D_fake = graph_nodes["D_fake"]

        # Applying softplus both to the generator and discriminator loss, seems to improve performances
        D_loss = tf.reduce_mean(tf.nn.softplus(D_real)) - tf.reduce_mean(tf.nn.softplus(D_fake))
        res = dict()
        res["D_loss"] = D_loss

        res_log = dict()
        res_log["D_main_loss"] = D_loss
        res_log["D_main_loss_fake"] = tf.reduce_mean(D_fake)
        res_log["D_main_loss_real"] = tf.reduce_mean(D_real)

        return res, res_log, dict()


class DCGANLossDiscriminatorBinaryCrossentropy(DCGANLossDiscriminator):

    def __init__(self, experiment, *args, **kwargs):
        super().__init__(experiment, *args, **kwargs)

        self.binary_cross_entropy_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def _forward(self, **graph_nodes):
        D_real = graph_nodes["D_real"]
        D_fake = graph_nodes["D_fake"]

        real_loss = self.binary_cross_entropy_loss(tf.zeros_like(D_real), D_real)
        fake_loss = self.binary_cross_entropy_loss(tf.ones_like(D_fake), D_fake)
        D_loss = real_loss + fake_loss
        res = dict()
        res["D_loss"] = D_loss

        res_log = dict()
        res_log["D_main_loss"] = D_loss
        res_log["D_main_loss_fake"] = tf.reduce_mean(D_fake)
        res_log["D_main_loss_real"] = tf.reduce_mean(D_real)

        return res, res_log, dict()


########################################################################################################################
#################################################### PIPES LOSSES ######################################################
########################################################################################################################

class PipesNumberLoss(_LevelLoss):
    """
    Requires that the number of tiles of type pipes generated by the network are between MIN_PIPES_NUMBER and MAX_PIPES_NUMBER
    Higher or lower values will receive a linear penalty.
    """

    def __init__(self, experiment, *args, **kwargs):
        super().__init__(experiment, *args, **kwargs)

        self.logger = self.experiment["LOGGER"]
        # parameters about the levels
        self.pipes_channels = [self.experiment["TILES_MAP"][x]['index'] for x in ['[', ']', '<', '>']]
        # retrieving the margins between which we want the number of tiles to be
        self.target_pipes_tile_number_min = self.experiment["PIPES_NUMBER_MIN"]
        self.target_pipes_tile_number_max = self.experiment["PIPES_NUMBER_MAX"]
        # getting usual data shape
        self.batch_size = self.experiment["BATCH_SIZE"]
        self.sample_height, self.sample_width, self.sample_channels = self.experiment["SHAPE"]
        # parameters about the control of the loss wrt the epoch number
        loss_from = experiment["PIPES_NUMBER_LOSS_FROM_EPOCH"]
        self.loss_from = loss_from if loss_from is not None else 0
        self.loss_incremental = bool(experiment["PIPES_NUMBER_LOSS_INCREMENTAL"])
        
    def _pre_processing(self, **graph_nodes):
        super()._pre_processing(**graph_nodes)
        self._requires(["G_probs", "current_epoch"], **graph_nodes)

    def _incremental_timing(self, graph_nodes, loss):
        """
        If experiment["PIPES_NUMBER_LOSS_INCREMENTAL"] is true then the loss is adjusted with a weight
        equal to (current epoch + 1 - epoch_from)/(total epochs + 1 - epoch_from), otherwise the loss is returned as it is.
        :param graph_nodes: the dict of important nodes
        :param loss: the loss that has to be controlled in time
        :return:
        """
        self.current_epoch = graph_nodes["current_epoch"]

        if self.loss_incremental:
            self.logger.info("Using incremental loss")
            weight = tf.cast(self.current_epoch + 1 - self.loss_from, tf.float32) / (self.experiment["LEARNING_EPOCHS"] + 1 - self.loss_from)
            return loss * weight
        else:
            self.logger.info("Not using incremental loss")
            return loss

    def _time_semantic_loss(self, graph_nodes, loss):
        """
        Get the loss as a tf node which value is zero if the current epoch is lower
        than the experiment["PIPES_NUMBER_LOSS_INCREMENTAL"] parameter, otherwise the loss is returned.
        :param graph_nodes: the dict of important nodes
        :param loss: the loss that has to be controlled in time
        :return:
        """
        self.current_epoch = graph_nodes["current_epoch"]

        self.logger.info("Using semantic loss starting from epoch %s" % self.loss_from)
        epoch_timed = tf.cond(
            self.current_epoch >= self.loss_from,
            true_fn=lambda: loss,
            false_fn=lambda: tf.constant(0.)
        )
        incremental_timed = self._incremental_timing(graph_nodes, epoch_timed)
        return incremental_timed

    def _forward(self, **graph_nodes):
        G_probs = graph_nodes["G_probs"]

        # reshaping G_probs from [batch_size, height, width, channels] to [batch_size, height * width, channels]
        G_probs = tf.reshape(G_probs, shape=(-1, self.sample_height * self.sample_width, self.sample_channels))

        # extract pipes channels
        G_pipes_probs = tf.gather(G_probs, self.pipes_channels, axis=-1)
        # G_pipes_probs has shape [batch_size, height * width, 4]

        # getting pipes probabilities per sample
        G_pipes_probs_per_sample = tf.reduce_sum(G_pipes_probs, axis=[-2, -1])
        # G_pipes_probs_per_sample has shape [batch_size]

        # gamma is the mean of target_pipes_tile_number_min and target_pipes_tile_number_max
        gamma = (self.target_pipes_tile_number_min + self.target_pipes_tile_number_max) / 2

        # teta is half the distance between target_pipes_tile_number_min and target_pipes_tile_number_max
        teta = (self.target_pipes_tile_number_max - self.target_pipes_tile_number_min) / 2

        # error as max(0, | actual - gamma | - teta)
        error = tf.math.maximum(tf.abs(G_pipes_probs_per_sample - gamma) - teta, tf.constant(0.0))

        # loss as mean of error over single samples
        loss = tf.reduce_mean(error)
        loss = self._time_semantic_loss(graph_nodes, loss)

        res = dict()
        res["G_loss"] = loss
        res["PipesNumberLoss"] = loss

        res_log = dict()
        res_log["PipesNumberLoss"] = loss

        return res, res_log, dict()


########################################################################################################################
################################################## CANNONS LOSSES ######################################################
########################################################################################################################


class CannonsNumberLoss(_LevelLoss):
    """
    Requires that the number of tiles of type pipes generated by the network are between MIN_PIPES_NUMBER and MAX_PIPES_NUMBER
    Higher or lower values will receive a linear penalty.
    """

    def __init__(self, experiment, *args, **kwargs):
        super().__init__(experiment, *args, **kwargs)

        self.logger = self.experiment["LOGGER"]
        # parameters about the levels
        self.cannons_channels = [self.experiment["TILES_MAP"][x]['index'] for x in ['b', 'B']]
        # retrieving the margins between which we want the number of tiles to be
        self.target_cannons_tile_number_min = self.experiment["CANNONS_NUMBER_MIN"]
        self.target_cannons_tile_number_max = self.experiment["CANNONS_NUMBER_MAX"]
        # getting usual data shape
        self.batch_size = self.experiment["BATCH_SIZE"]
        self.sample_height, self.sample_width, self.sample_channels = self.experiment["SHAPE"]
        # parameters about the control of the loss wrt the epoch number
        loss_from = experiment["CANNONS_NUMBER_LOSS_FROM_EPOCH"]
        self.loss_from = loss_from if loss_from is not None else 0
        self.loss_incremental = bool(experiment["CANNONS_NUMBER_LOSS_INCREMENTAL"])
        
    def _pre_processing(self, **graph_nodes):
        super()._pre_processing(**graph_nodes)
        self._requires(["G_probs", "current_epoch"], **graph_nodes)

    def _incremental_timing(self, graph_nodes, loss):
        """
        If experiment["CANNONS_NUMBER_LOSS_INCREMENTAL"] is true then the loss is adjusted with a weight
        equal to (current epoch + 1 - epoch_from)/(total epochs + 1 - epoch_from), otherwise the loss is returned as it is.
        :param graph_nodes: the dict of important nodes
        :param loss: the loss that has to be controlled in time
        :return:
        """
        self.current_epoch = graph_nodes["current_epoch"]

        if self.loss_incremental:
            self.logger.info("Using incremental loss")
            weight = tf.cast(self.current_epoch + 1 - self.loss_from, tf.float32) / (self.experiment["LEARNING_EPOCHS"] + 1 - self.loss_from)
            return loss * weight
        else:
            self.logger.info("Not using incremental loss")
            return loss

    def _time_semantic_loss(self, graph_nodes, loss):
        """
        Get the loss as a tf node which value is zero if the current epoch is lower
        than the experiment["CANNONS_NUMBER_LOSS_INCREMENTAL"] parameter, otherwise the loss is returned.
        :param graph_nodes: the dict of important nodes
        :param loss: the loss that has to be controlled in time
        :return:
        """
        self.current_epoch = graph_nodes["current_epoch"]

        self.logger.info("Using semantic loss starting from epoch %s" % self.loss_from)
        epoch_timed = tf.cond(
            self.current_epoch >= self.loss_from,
            true_fn=lambda: loss,
            false_fn=lambda: tf.constant(0.)
        )
        incremental_timed = self._incremental_timing(graph_nodes, epoch_timed)
        return incremental_timed

    def _forward(self, **graph_nodes):
        G_probs = graph_nodes["G_probs"]

        # reshaping G_probs from [batch_size, height, width, channels] to [batch_size, height * width, channels]
        G_probs = tf.reshape(G_probs, shape=(-1, self.sample_height * self.sample_width, self.sample_channels))

        # extract cannons channels
        G_cannons_probs = tf.gather(G_probs, self.cannons_channels, axis=-1)
        # G_cannons_probs has shape [batch_size, height * width, 2]

        # getting cannons probabilities per sample
        G_cannons_probs_per_sample = tf.reduce_sum(G_cannons_probs, axis=[-2, -1])
        # G_cannons_probs_per_sample has shape [batch_size]

        # gamma is the mean of target_cannons_tile_number_min and target_cannons_tile_number_max
        gamma = (self.target_cannons_tile_number_min + self.target_cannons_tile_number_max) / 2

        # teta is half the distance between target_cannons_tile_number_min and target_cannons_tile_number_max
        teta = (self.target_cannons_tile_number_max - self.target_cannons_tile_number_min) / 2

        # error as max(0, | actual - gamma | - teta)
        error = tf.math.maximum(tf.abs(G_cannons_probs_per_sample - gamma) - teta, tf.constant(0.0))

        # loss as mean of error over single samples
        loss = tf.reduce_mean(error)
        loss = self._time_semantic_loss(graph_nodes, loss)

        res = dict()
        res["G_loss"] = loss
        res["CannonsNumberLoss"] = loss

        res_log = dict()
        res_log["CannonsNumberLoss"] = loss

        return res, res_log, dict()


########################################################################################################################
################################################## OTHER LOSSES ########################################################
########################################################################################################################

"""
Implementing the reachability as a loss, trying to compute all moves as a set of tensorflow functions and keeping them differentiable.
"""

'''
class DCGANRechabilityLoss(_LevelLoss):

    def __init__(self, experiment):
        super().__init__(experiment)

        # define Super Mario Bros possible moves        
        kernel_height = 11
        kernel_width = 13

        # custom kernel for Super Mario Bros moves
        kernel = np.ones((kernel_height, kernel_width), dtype=float)
        for i in range(kernel_height):
            for j in range(kernel_width):
                #if j > (kernel_width - 1) // 2:
                #    kernel[i, j] = .0
                if - i + j <= - 6 or i + j >= 18:
                    kernel[i, j] = .0

        kernel = np.expand_dims(kernel, axis=2)
        kernel = np.expand_dims(kernel, axis=3)
        
        init = tf.constant_initializer(kernel)
        self.conv2d = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=kernel.shape[:2],
            strides=1,
            padding='same',
            data_format='channels_last',
            kernel_initializer=init,
            use_bias=False
        )
        # avoid weights change
        self.conv2d.trainable = False

        # paddings
        self.paddings_up = tf.constant([[0, 0], [1, 0], [0, 0]])
        self.paddings_down = tf.constant([[0, 0], [0, 1], [0, 0]])

    def _pre_processing(self, **graph_nodes):
        assert "G_output_logit" in graph_nodes

    def _forward(self, **graph_nodes):
        g_output_logit = tf.nn.softmax(graph_nodes["G_output_logit"])
        reachability = self._reachability(g_output_logit, graph_nodes["z"])

        res = dict()
        # there should be a small bug that is ruining the training, using a 0 loss at the moment
        res["G_loss"] = tf.constant(.0)
        #res["G_loss"] = 1 - tf.reduce_mean(reachability)
        # G_rechability used only for statistics
        # 1 - tf.reduce_mean(rechability)
        res["G_reachability"] = tf.reduce_mean(reachability)
        
        return res, dict(), dict()

    def _reachability(self, level, z):
        """
        The reachability constraint is derivable, then is momentarily used as a loss
        """
        level = tf.reshape(level, [-1] + level.get_shape().as_list()[-3:])

        # shape (bs, height, width, channels)
        # get probability of being a passable tile over all the possible tiles
        # indices = tf.constant([2, 5, 10])
        indices = tf.constant([2, 5])
        input_level = tf.math.divide(tf.reduce_sum(tf.gather(level, indices, axis=-1), axis=-1), tf.reduce_sum(level, axis=-1))
        
        """
        # get some levels to make experiments
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        level = sess.run(input_level, feed_dict={z: np.random.normal(
            loc=0, scale=1.0, size=(2, 32))})
        np.save('./level', level)
        exit()
        """
        
        # shape (bs, height, width)
        level_height = input_level.shape[1]
        level_width = input_level.shape[2]

        ################# RECHABILITY OF MOVES ####################

        # define the shadow matrix, the level matrix with the rows shifted
        # up and padded with 0. Will help identifiyng stationary tiles
        shadow = tf.pad(1 - input_level[:, 1:, :], self.paddings_down, mode='CONSTANT', constant_values=.0)

        # create matrix of probabilities to be a stationary point.
        # A stationary point is a tile with a high probability to be 
        # air-like that stand over a tile with a high probability
        # of being solid.
        stationary = tf.math.multiply(input_level, shadow)
        # normalize such that best couples air-solid keeps their best value
        #stationary = stationary / tf.sqrt(tf.reduce_max(stationary))

        # use the convolution to find all reachables points from the stationary points
        reachable = tf.expand_dims(stationary, axis=-1)
        reachable = self.conv2d(reachable)
        reachable = tf.squeeze(reachable, axis=-1)

        # values to interval 0 - mean
        # this should avoid having big differences between tiles that can be 
        # reached from lot of points with tiles that can be be reached from
        # only a few points. it is only important that a tile is reachable
        #reachable = tf.clip_by_value(reachable, 0, tf.reduce_mean(reachable))

        # normalize
        #reachable = tf.clip_by_value(reachable, 0, tf.reduce_mean(reachable))
        reachable = tf.tanh(reachable)
        # lower values where air is less probable
        reachable = tf.math.multiply(reachable, input_level)


        # simulate gravity: reachable air-like tiles will expand rechability to the bottom
        def body(i, reachable):
            previous_reachable = reachable[:, i-1:i, :]
            actual_reachable = reachable[:, i:i+1, :]
            input_row = input_level[:, i:i+1, :]

            new_row = tf.math.multiply(previous_reachable, input_row)
            new_row = new_row / tf.sqrt(tf.reduce_max(new_row))
            new_row = tf.math.maximum(new_row, actual_reachable)

            res = tf.concat(
                [reachable[:, :i, :], new_row, reachable[:, i+1:, :]],
                axis=1,
            )
            return tf.add(i, 1), res

        def cond(i, reachable):
            return tf.less(i, level_height)

        out_shape = reachable.get_shape().as_list()
        out_shape[1] = None
        _, reachable = tf.while_loop(cond, body, [tf.constant(1), reachable], 
                                    shape_invariants=[tf.TensorShape([]), tf.TensorShape(out_shape)])

        # debugging
        #b = tf.reduce_max(reachable)
        reachable = tf.concat(
            [input_level[:, :, 0:1], reachable[:, :, 1:]],
            axis=2,
        )

        # final rechability: from left to right
        def cond(i, reachable):
            return tf.less(i, level_width)

        def body(i, reachable):
            previous_reachable_column = reachable[:, :, i-1:i]
            actual_reachable_column = reachable[:, :, i:i+1]

            new_column = tf.math.multiply(previous_reachable_column, actual_reachable_column)
            new_column = new_column / tf.sqrt(tf.reduce_max(new_column))

            def cond2(i, column):
                return tf.less(i, level_height - 1)

            # jumps
            def body_jump(i, column):
                mult = tf.pad(column[:, :-1, :], self.paddings_up, mode='CONSTANT', constant_values=.0)
                movement = tf.math.multiply(mult, actual_reachable_column)
                movement = movement / tf.sqrt(tf.reduce_max(movement))
                return tf.add(i, 1), tf.math.maximum(column, movement)

            # falls
            def body_fall(i, column):
                mult = tf.pad(column[:, 1:, :], self.paddings_down, mode='CONSTANT', constant_values=.0)
                movement = tf.math.multiply(mult, actual_reachable_column)
                movement = movement / tf.sqrt(tf.reduce_max(movement))
                return tf.add(i, 1), tf.math.maximum(column, movement)

            _, new_column = tf.while_loop(cond2, body_jump, [tf.constant(0), new_column])
            _, new_column = tf.while_loop(cond2, body_fall, [tf.constant(0), new_column])

            # normalize
            # using sqrt such that high values like 0.9 * 0.9 = 0.81 becomes newly 0.9 after multiplication
            #new_column = new_column / tf.sqrt(tf.reduce_max(new_column))

            # set reachable
            reachable = tf.concat(
                [reachable[:, :, :i], new_column, reachable[:, :, i+1:]],
                axis=2,
            )
            return tf.add(i, 1), reachable

        out_shape = reachable.get_shape().as_list()
        out_shape[2] = None
        _, reachable = tf.while_loop(cond, body, [tf.constant(1), reachable], shape_invariants=[tf.TensorShape([]), tf.TensorShape(out_shape)])

        #c = tf.reduce_max(reachable)
        # taking the most reachable tile of the last column
        return tf.math.reduce_max(reachable[:, :, -1], axis=-1)
'''