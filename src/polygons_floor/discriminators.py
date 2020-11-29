"""
File containing the polygons/floor discriminators architectures.
"""
import numpy as np
import tensorflow as tf

from base_layers import Discriminator

from tensorflow.contrib.layers import xavier_initializer as xavier_init
from functools import partial
from tensorflow.keras.layers import Layer, Dense, Conv2D, LeakyReLU
from computables import ConstraintsComputable


# some architectures (and hyper-parameters) are drawn from the original work on
# "Boundary-Seeking Generative Adversarial Networks"
# ref: https://github.com/rdevon/BGAN
# The original code was written in Theano/Lasagne and has been ported to TF

# NOTE: conv2d/deconv2d paddings may be slightly different from the ones in
# Lasagne. To make them equivalent, a manual pad2 is necessary. Sketch:
# pad2 = lambda x: tf.pad(x, [[0, 0], [2, 2], [2, 2], [0, 0]], "CONSTANT")

# NOTE: this batch normalization is slightly different from the one in Lasagne.
# See the LS-TF porting notes for further info.


class PolygonsFloorDiscriminator(Discriminator):

    def __init__(self, experiment):
        super().__init__(experiment)
        self.h_dim = experiment["H_DIM"]
        self.logger = experiment["LOGGER"]
        self.shape_G_sample = [-1] + experiment["SHAPE"]

        leakiness = 0.01 if not experiment["LEAKINESS"] or experiment["LEAKINESS"] < 0 else experiment["LEAKINESS"]
        self.leaky_relu = LeakyReLU(alpha=leakiness)

        self.use_constraints = tf.placeholder(tf.bool, shape=())

    def _pre_processing(self, **graph_nodes):
        assert "G_sample" in graph_nodes

    def _forward(self, **graph_nodes):
        D_real = self._network(self.X, False)

        # from (batch, num bgan samples, shape) to (-1, shape)
        G_sample = tf.reshape(graph_nodes["G_sample"], self.shape_G_sample)
        D_fake = self._network(G_sample, True)

        nodes = dict()
        nodes["discriminator"] = self
        nodes["D_real"] = D_real
        nodes["D_fake"] = D_fake
        nodes["use_constraints"] = self.use_constraints
        nodes["X"] = self.X

        return nodes, dict(), dict()

    def _network(self, node, constraints_placeholder):
        raise NotImplementedError("Should be overridden by subclasses")


########################################################################################################################
########################################### GAN ARCHITECTURES ##########################################################
########################################################################################################################

class Gan16Discriminator(PolygonsFloorDiscriminator):

    def __init__(self, experiment):
        super().__init__(experiment)

        with tf.variable_scope("discriminator"):
            with tf.variable_scope("hidden1"):
                self.conv2d_1 = Conv2D(filters=self.h_dim, kernel_size=5, strides=2, padding="same",
                                       activation=self.leaky_relu, kernel_initializer=xavier_init())
            with tf.variable_scope("hidden2"):
                self.conv2d_2 = Conv2D(filters=self.h_dim * 2, kernel_size=5, strides=2, padding="same",
                                       activation=self.leaky_relu, kernel_initializer=xavier_init())
            with tf.variable_scope("hidden3"):
                self.dense_3 = Dense(1024, activation=self.leaky_relu, kernel_initializer=xavier_init())
            with tf.variable_scope("output"):
                self.dense_output = Dense(1, kernel_initializer=xavier_init())

    def _network(self, node, reuse):
        msg = "D_SHAPE {} {} [reuse={}]"

        with tf.variable_scope("discriminator", reuse=reuse):
            reuse = bool(tf.AUTO_REUSE)
            self.logger.debug(msg.format("in", node.shape, reuse))

            with tf.variable_scope("hidden1"):
                d_hidden1 = self.conv2d_1(node)
                self.logger.debug(msg.format("dh1", d_hidden1.shape, reuse))

            with tf.variable_scope("hidden2"):
                d_hidden2 = self.conv2d_2(d_hidden1)
                self.logger.debug(msg.format("dh2", d_hidden2.shape, reuse))
                d_hidden2 = tf.reshape(d_hidden2, [-1, np.prod(d_hidden2.shape[1:], dtype=int)])
                self.logger.debug(msg.format("dh2", d_hidden2.shape, reuse))

            with tf.variable_scope("hidden3"):
                d_hidden3 = self.dense_3(d_hidden2)
                self.logger.debug(msg.format("dh3", d_hidden3.shape, reuse))

            with tf.variable_scope("output"):
                d_out = self.dense_output(d_hidden3)
                self.logger.debug(msg.format("out", d_out.shape, reuse))

        return d_out


class Gan16Discriminator32Layer(PolygonsFloorDiscriminator):

    def __init__(self, experiment):
        super().__init__(experiment)

        with tf.variable_scope("discriminator"):
            with tf.variable_scope("hidden1"):
                self.conv2d_1 = Conv2D(filters=self.h_dim, kernel_size=5, strides=2, padding="same",
                                       activation=self.leaky_relu, kernel_initializer=xavier_init())
            with tf.variable_scope("hidden2"):
                self.conv2d_2 = Conv2D(filters=self.h_dim * 2, kernel_size=5, strides=2, padding="same",
                                       activation=self.leaky_relu, kernel_initializer=xavier_init())
            with tf.variable_scope("hidden3"):
                self.dense_3 = Dense(1024, activation=self.leaky_relu, kernel_initializer=xavier_init())
            with tf.variable_scope("hidden4"):
                self.dense_4 = Dense(32, activation=self.leaky_relu, kernel_initializer=xavier_init())
            with tf.variable_scope("output"):
                self.dense_output = Dense(1, kernel_initializer=xavier_init())

    def _network(self, node, reuse):
        msg = "D_SHAPE {} {} [reuse={}]"

        with tf.variable_scope("discriminator", reuse=reuse):
            reuse = bool(tf.AUTO_REUSE)
            self.logger.debug(msg.format("in", node.shape, reuse))

            with tf.variable_scope("hidden1"):
                d_hidden1 = self.conv2d_1(node)
                self.logger.debug(msg.format("dh1", d_hidden1.shape, reuse))

            with tf.variable_scope("hidden2"):
                d_hidden2 = self.conv2d_2(d_hidden1)
                self.logger.debug(msg.format("dh2", d_hidden2.shape, reuse))
                d_hidden2 = tf.reshape(d_hidden2, [-1, np.prod(d_hidden2.shape[1:], dtype=int)])
                self.logger.debug(msg.format("dh2", d_hidden2.shape, reuse))

            with tf.variable_scope("hidden3"):
                d_hidden3 = self.dense_3(d_hidden2)
                self.logger.debug(msg.format("dh3", d_hidden3.shape, reuse))

            with tf.variable_scope("hidden4"):
                d_hidden4 = self.dense_4(d_hidden3)
                self.logger.debug(msg.format("dh4", d_hidden4.shape, reuse))

            with tf.variable_scope("output"):
                d_out = self.dense_output(d_hidden4)
                self.logger.debug(msg.format("out", d_out.shape, reuse))

        return d_out


Gan20Discriminator = Gan16Discriminator
Gan20Discriminator32Layer = Gan16Discriminator32Layer

Gan28Discriminator = Gan16Discriminator
Gan28Discriminator32Layer = Gan16Discriminator32Layer


class Gan32Discriminator(PolygonsFloorDiscriminator):

    def __init__(self, experiment):
        super().__init__(experiment)

        with tf.variable_scope("discriminator"):
            with tf.variable_scope("hidden1"):
                self.conv2d_1 = Conv2D(filters=self.h_dim, kernel_size=5, strides=2, padding="same",
                                       activation=self.leaky_relu, kernel_initializer=xavier_init())
            with tf.variable_scope("hidden2"):
                self.conv2d_2 = Conv2D(filters=self.h_dim * 2, kernel_size=5, strides=2, padding="same",
                                       activation=self.leaky_relu, kernel_initializer=xavier_init())
            with tf.variable_scope("hidden3"):
                self.conv2d_3 = Conv2D(filters=self.h_dim * 4, kernel_size=5, strides=2,
                                       padding="same", activation=self.leaky_relu, kernel_initializer=xavier_init())
            with tf.variable_scope("output"):
                self.dense_output = Dense(1, kernel_initializer=xavier_init())

    def _network(self, node, reuse):
        msg = "D_SHAPE {} {} [reuse={}]"

        with tf.variable_scope("discriminator", reuse=reuse):
            reuse = bool(tf.AUTO_REUSE)
            self.logger.debug(msg.format("in", node.shape, reuse))

            with tf.variable_scope("hidden1"):
                d_hidden1 = self.conv2d_1(node)
                self.logger.debug(msg.format("dh1", d_hidden1.shape, reuse))

            with tf.variable_scope("hidden2"):
                d_hidden2 = self.conv2d_2(d_hidden1)
                self.logger.debug(msg.format("dh2", d_hidden2.shape, reuse))

            with tf.variable_scope("hidden3"):
                d_hidden3 = self.conv2d_3(d_hidden2)
                self.logger.debug(msg.format("dh3", d_hidden3.shape, reuse))
                d_hidden3 = tf.reshape(d_hidden3, [-1, np.prod(d_hidden3.shape[1:], dtype=int)])
                self.logger.debug(msg.format("dh3", d_hidden3.shape, reuse))

            with tf.variable_scope("output"):
                d_out = self.dense_output(d_hidden3)
                self.logger.debug(msg.format("out", d_out.shape, reuse))

        return d_out


class Gan32Discriminator32Layer(PolygonsFloorDiscriminator):

    def __init__(self, experiment):
        super().__init__(experiment)

        with tf.variable_scope("discriminator"):
            with tf.variable_scope("hidden1"):
                self.conv2d_1 = Conv2D(filters=self.h_dim, kernel_size=5, strides=2, padding="same",
                                       activation=self.leaky_relu, kernel_initializer=xavier_init())
            with tf.variable_scope("hidden2"):
                self.conv2d_2 = Conv2D(filters=self.h_dim * 2, kernel_size=5, strides=2, padding="same",
                                       activation=self.leaky_relu, kernel_initializer=xavier_init())
            with tf.variable_scope("hidden3"):
                self.conv2d_3 = Conv2D(filters=self.h_dim * 4, kernel_size=5, strides=2,
                                       padding="same", activation=self.leaky_relu, kernel_initializer=xavier_init())
            with tf.variable_scope("hidden4"):
                self.dense_4 = Dense(32, activation=self.leaky_relu, kernel_initializer=xavier_init())

            with tf.variable_scope("output"):
                self.dense_output = Dense(1, kernel_initializer=xavier_init())

    def _network(self, node, reuse):
        msg = "D_SHAPE {} {} [reuse={}]"

        with tf.variable_scope("discriminator", reuse=reuse):
            reuse = bool(tf.AUTO_REUSE)
            self.logger.debug(msg.format("in", node.shape, reuse))

            with tf.variable_scope("hidden1"):
                d_hidden1 = self.conv2d_1(node)
                self.logger.debug(msg.format("dh1", d_hidden1.shape, reuse))

            with tf.variable_scope("hidden2"):
                d_hidden2 = self.conv2d_2(d_hidden1)
                self.logger.debug(msg.format("dh2", d_hidden2.shape, reuse))

            with tf.variable_scope("hidden3"):
                d_hidden3 = self.conv2d_3(d_hidden2)
                self.logger.debug(msg.format("dh3", d_hidden3.shape, reuse))
                d_hidden3 = tf.reshape(d_hidden3, [-1, np.prod(d_hidden3.shape[1:], dtype=int)])
                self.logger.debug(msg.format("dh3", d_hidden3.shape, reuse))

            with tf.variable_scope("hidden4"):
                d_hidden4 = self.dense_4(d_hidden3)
                self.logger.debug(msg.format("dh4", d_hidden4.shape, reuse))

            with tf.variable_scope("output"):
                d_out = self.dense_output(d_hidden3)
                self.logger.debug(msg.format("out", d_out.shape, reuse))

        return d_out


class Gan60Discriminator32Layer(PolygonsFloorDiscriminator):

    def __init__(self, experiment):
        super().__init__(experiment)

        with tf.variable_scope("discriminator"):
            with tf.variable_scope("hidden1"):
                self.conv2d_1 = Conv2D(filters=self.h_dim, kernel_size=5, strides=2, padding="same",
                                       activation=self.leaky_relu, kernel_initializer=xavier_init())
            with tf.variable_scope("hidden15"):
                self.conv2d_15 = Conv2D(filters=self.h_dim, kernel_size=5, strides=2, padding="same",
                                        activation=self.leaky_relu, kernel_initializer=xavier_init())
            with tf.variable_scope("hidden2"):
                self.conv2d_2 = Conv2D(filters=self.h_dim * 2, kernel_size=5, strides=2, padding="same",
                                       activation=self.leaky_relu, kernel_initializer=xavier_init())
            with tf.variable_scope("hidden3"):
                self.conv2d_3 = Dense(1024, activation=self.leaky_relu, kernel_initializer=xavier_init())

            with tf.variable_scope("hidden4"):
                self.dense_4 = Dense(32, activation=self.leaky_relu, kernel_initializer=xavier_init())

            with tf.variable_scope("output"):
                self.dense_output = Dense(1, kernel_initializer=xavier_init())

    def _network(self, node, reuse):
        msg = "D_SHAPE {} {} [reuse={}]"

        with tf.variable_scope("discriminator", reuse=reuse):
            reuse = bool(tf.AUTO_REUSE)
            self.logger.debug(msg.format("in", node.shape, reuse))

            with tf.variable_scope("hidden1"):
                d_hidden1 = self.conv2d_1(node)
                self.logger.debug(msg.format("dh1", d_hidden1.shape, reuse))

            with tf.variable_scope("hidden15"):
                d_hidden15 = self.conv2d_15(d_hidden1)
                self.logger.debug(msg.format("dh15", d_hidden15.shape, reuse))

            with tf.variable_scope("hidden2"):
                d_hidden2 = self.conv2d_2(d_hidden15)
                self.logger.debug(msg.format("dh2", d_hidden2.shape, reuse))
                d_hidden2 = tf.reshape(d_hidden2, [-1, np.prod(d_hidden2.shape[1:], dtype=int)])
                self.logger.debug(msg.format("dh2", d_hidden2.shape, reuse))

            with tf.variable_scope("hidden3"):
                d_hidden3 = self.conv2d_3(d_hidden2)
                self.logger.debug(msg.format("dh3", d_hidden3.shape, reuse))

            with tf.variable_scope("hidden4"):
                d_hidden4 = self.dense_4(d_hidden3)
                self.logger.debug(msg.format("dh4", d_hidden4.shape, reuse))

            with tf.variable_scope("output"):
                d_out = self.dense_output(d_hidden3)
                self.logger.debug(msg.format("out", d_out.shape, reuse))

        return d_out


########################################################################################################################
########################################### CAN ARCHITECTURES ##########################################################
########################################################################################################################


class CanPolygonsFloorDiscriminator(PolygonsFloorDiscriminator):

    def __init__(self, experiment):
        super().__init__(experiment)
        self.total_constraints = sum([len(Computable.constraints_names()) for Computable in experiment["COMPUTABLES"] if
                                      issubclass(Computable, ConstraintsComputable)])
        self.C_real = tf.placeholder(tf.float32, [None, self.total_constraints], "C_real_placeholder")
        self.C_fake = tf.placeholder(tf.float32, [None, self.total_constraints], "C_fake_placeholder")
        self.use_constraints = tf.placeholder(tf.bool, shape=())

    def _log_shape(self, name, shape, reuse):
        msg = "D_SHAPE {} {} [reuse={}]"
        self.logger.debug(msg.format(name, shape, reuse))

    def _forward(self, **graph_nodes):
        D_out_real = self._network(self.X, self.C_real, False)

        # from (batch, num_bgan_samples, shape) to (-1, shape)
        G_sample = tf.reshape(graph_nodes["G_sample"], self.shape_G_sample)
        D_out_fake = self._network(G_sample, self.C_fake, True)

        nodes = dict()
        # d_out
        nodes["D_real"] = D_out_real["D_out"]
        nodes["D_fake"] = D_out_fake["D_out"]

        to_add = ["D_hidden4", "D_hidden4_sigmoid", "D_out_kernel", "D_constraints_kernel"]
        for name in to_add:
            if name in D_out_real:
                nodes[name + "_real"] = D_out_real[name]
            if name in D_out_fake:
                nodes[name + "_fake"] = D_out_fake[name]

        # placeholders for constraints
        nodes["discriminator"] = self
        nodes["C_real"] = self.C_real
        nodes["C_fake"] = self.C_fake
        nodes["use_constraints"] = self.use_constraints
        nodes["X"] = self.X

        return nodes, dict(), dict()

    def _network(self, node, constraints_placeholder, reuse):
        raise NotImplementedError("Should be overridden by subclasses")


class Can16Discriminator32LayerAuto(CanPolygonsFloorDiscriminator):

    def __init__(self, experiment):
        super().__init__(experiment)

        with tf.variable_scope("discriminator"):
            with tf.variable_scope("hidden1"):
                self.conv2d_1 = Conv2D(filters=self.h_dim, kernel_size=5, strides=2, padding="same",
                                       activation=self.leaky_relu, kernel_initializer=xavier_init())
            with tf.variable_scope("hidden2"):
                self.conv2d_2 = Conv2D(filters=self.h_dim * 2, kernel_size=5, strides=2, padding="same",
                                       activation=self.leaky_relu, kernel_initializer=xavier_init())
            with tf.variable_scope("hidden3"):
                self.dense_3 = Dense(1024, activation=self.leaky_relu, kernel_initializer=xavier_init())
            with tf.variable_scope("hidden4"):
                self.dense_4 = Dense(32, activation=self.leaky_relu, kernel_initializer=xavier_init())
            with tf.variable_scope("shared_weights"):
                self.d_out_kernel = tf.get_variable("d_out_kernel", shape=[32, 1], initializer=xavier_init())
                self.d_out_bias = tf.get_variable("d_out_bias", shape=[1, 1], initializer=xavier_init())
            with tf.variable_scope("constrained_out"):
                self.d_constraints_kernel = tf.get_variable("d_constraints_kernel", shape=[self.total_constraints, 1],
                                                            initializer=xavier_init())

    def _skip_constraints(self, res, d_hidden4, reuse):
        with tf.variable_scope("output"):
            d_out = tf.add(tf.matmul(d_hidden4, self.d_out_kernel), self.d_out_bias, name="d_out_{}".format(reuse))
            self._log_shape("out", d_out.shape, reuse)

            return d_out

    def _apply_constraints(self, res, d_hidden4, constraints_placeholder, reuse):
        with tf.variable_scope("constrained_out"):
            self._log_shape("d_constraints_kernel", self.d_constraints_kernel.shape, reuse)
            input_concat = tf.concat([d_hidden4, constraints_placeholder], axis=1, name="input_concat_{}".format(reuse))
            self._log_shape("input_concat", input_concat.shape, reuse)
            weight_concat = tf.concat([self.d_out_kernel, self.d_constraints_kernel],
                                      axis=0, name="weight_concat_{}".format(reuse))
            self._log_shape("weight_concat", weight_concat.shape, reuse)
            d_constrained_out = tf.add(tf.matmul(input_concat, weight_concat), self.d_out_bias,
                                       name="d_constrained_out_{}".format(reuse))
            self._log_shape("constrained_out", d_constrained_out.shape, reuse)

            res["D_out_kernel"] = self.d_out_kernel
            res["D_constraints_kernel"] = self.d_constraints_kernel
            return d_constrained_out

    def _network(self, node, constraints_placeholder, reuse):
        with tf.variable_scope("discriminator", reuse=reuse):
            reuse = bool(tf.AUTO_REUSE)
            self._log_shape("in", node.shape, reuse)

            with tf.variable_scope("hidden1"):
                d_hidden1 = self.conv2d_1(node)
                self._log_shape("dh1", d_hidden1.shape, reuse)

            with tf.variable_scope("hidden2"):
                d_hidden2 = self.conv2d_2(d_hidden1)
                self._log_shape("dh2", d_hidden2.shape, reuse)
                d_hidden2 = tf.reshape(d_hidden2, [-1, np.prod(d_hidden2.shape[1:], dtype=int)])
                self._log_shape("dh2", d_hidden2.shape, reuse)

            with tf.variable_scope("hidden3"):
                d_hidden3 = self.dense_3(d_hidden2)
                self._log_shape("dh3", d_hidden3.shape, reuse)

            with tf.variable_scope("hidden4"):
                d_hidden4 = self.dense_4(d_hidden3)
                self._log_shape("dh4", d_hidden4.shape, reuse)

            self._log_shape("d_out_kernel", self.d_out_kernel.shape, reuse)
            self._log_shape("d_out_bias", self.d_out_bias.shape, reuse)

            res = dict()
            d_out = tf.cond(self.use_constraints,
                            false_fn=lambda: self._skip_constraints(res, d_hidden4, reuse),
                            true_fn=lambda: self._apply_constraints(res, d_hidden4, constraints_placeholder, reuse))

            res["D_out"] = d_out
            res["D_hidden4"] = d_hidden4
            return res


Can20Discriminator32LayerAuto = Can16Discriminator32LayerAuto


class Can28Discriminator32LayerAuto(CanPolygonsFloorDiscriminator):

    def __init__(self, experiment):
        super().__init__(experiment)

        with tf.variable_scope("discriminator"):
            with tf.variable_scope("hidden1"):
                self.conv2d_1 = Conv2D(filters=self.h_dim * 4, kernel_size=3, strides=1, padding="same",
                                       activation=self.leaky_relu, kernel_initializer=xavier_init())
            with tf.variable_scope("hidden2"):
                self.conv2d_2 = Conv2D(filters=self.h_dim, kernel_size=4, strides=2, padding="same",
                                       activation=self.leaky_relu, kernel_initializer=xavier_init())
            with tf.variable_scope("hidden3"):
                self.dense_3 = Dense(self.h_dim * 7 * 7, activation=self.leaky_relu, kernel_initializer=xavier_init())
            with tf.variable_scope("hidden4"):
                self.dense_4 = Dense(self.h_dim * 16, activation=self.leaky_relu, kernel_initializer=xavier_init())
            with tf.variable_scope("shared_weights"):
                self.d_out_kernel = tf.get_variable("d_out_kernel", shape=[self.h_dim * 16, 1],
                                                    initializer=xavier_init())
                self.d_out_bias = tf.get_variable("d_out_bias", shape=[1, 1], initializer=xavier_init())
            with tf.variable_scope("constrained_out"):
                self.d_constraints_kernel = tf.get_variable("d_constraints_kernel", shape=[self.total_constraints, 1],
                                                            initializer=xavier_init())

    def _skip_constraints(self, res, d_hidden4, reuse):
        with tf.variable_scope("output"):
            d_out = tf.add(tf.matmul(d_hidden4, self.d_out_kernel), self.d_out_bias, name="d_out_{}".format(reuse))
            self._log_shape("out", d_out.shape, reuse)

            return d_out

    def _apply_constraints(self, res, d_hidden4, constraints_placeholder, reuse):
        with tf.variable_scope("constrained_out"):
            self._log_shape("d_constraints_kernel", self.d_constraints_kernel.shape, reuse)
            input_concat = tf.concat([d_hidden4, constraints_placeholder], axis=1, name="input_concat_{}".format(reuse))
            self._log_shape("input_concat", input_concat.shape, reuse)
            weight_concat = tf.concat([self.d_out_kernel, self.d_constraints_kernel],
                                      axis=0, name="weight_concat_{}".format(reuse))
            self._log_shape("weight_concat", weight_concat.shape, reuse)
            d_constrained_out = tf.add(tf.matmul(input_concat, weight_concat), self.d_out_bias,
                                       name="d_constrained_out_{}".format(reuse))
            self._log_shape("constrained_out", d_constrained_out.shape, reuse)

            res["D_out_kernel"] = self.d_out_kernel
            res["D_constraints_kernel"] = self.d_constraints_kernel
            return d_constrained_out

    def _network(self, node, constraints_placeholder, reuse):
        with tf.variable_scope("discriminator", reuse=reuse):
            reuse = bool(tf.AUTO_REUSE)
            self._log_shape("in", node.shape, reuse)

            with tf.variable_scope("hidden1"):
                d_hidden1 = self.conv2d_1(node)
                self._log_shape("dh1", d_hidden1.shape, reuse)

            with tf.variable_scope("hidden2"):
                d_hidden2 = self.conv2d_2(d_hidden1)
                self._log_shape("dh2", d_hidden2.shape, reuse)
                d_hidden2 = tf.reshape(d_hidden2, [-1, np.prod(d_hidden2.shape[1:], dtype=int)])
                self._log_shape("dh2", d_hidden2.shape, reuse)

            with tf.variable_scope("hidden3"):
                d_hidden3 = self.dense_3(d_hidden2)
                self._log_shape("dh3", d_hidden3.shape, reuse)

            with tf.variable_scope("hidden4"):
                d_hidden4 = self.dense_4(d_hidden3)
                self._log_shape("dh4", d_hidden4.shape, reuse)

            self._log_shape("d_out_kernel", self.d_out_kernel.shape, reuse)
            self._log_shape("d_out_bias", self.d_out_bias.shape, reuse)

            res = dict()
            d_out = tf.cond(self.use_constraints,
                            false_fn=lambda: self._skip_constraints(res, d_hidden4, reuse),
                            true_fn=lambda: self._apply_constraints(res, d_hidden4, constraints_placeholder, reuse))

            res["D_out"] = d_out
            res["D_hidden4"] = d_hidden4
            return res


class Can20Discriminator32LayerSigmoidAuto(CanPolygonsFloorDiscriminator):

    def __init__(self, experiment):
        super().__init__(experiment)

        with tf.variable_scope("discriminator"):
            with tf.variable_scope("hidden1"):
                self.conv2d_1 = Conv2D(filters=self.h_dim, kernel_size=5, strides=2, padding="same",
                                       activation=self.leaky_relu, kernel_initializer=xavier_init())
            with tf.variable_scope("hidden2"):
                self.conv2d_2 = Conv2D(filters=self.h_dim * 2, kernel_size=5, strides=2, padding="same",
                                       activation=self.leaky_relu, kernel_initializer=xavier_init())
            with tf.variable_scope("hidden3"):
                self.dense_3 = Dense(1024, activation=self.leaky_relu, kernel_initializer=xavier_init())
            with tf.variable_scope("hidden4"):
                self.dense_4 = Dense(32, activation=self.leaky_relu, kernel_initializer=xavier_init())
            with tf.variable_scope("shared_weights"):
                self.d_out_kernel = tf.get_variable("d_out_kernel", shape=[32, 1], initializer=xavier_init())
                self.d_out_bias = tf.get_variable("d_out_bias", shape=[1, 1], initializer=xavier_init())
            with tf.variable_scope("constrained_out"):
                self.d_constraints_kernel = tf.get_variable("d_constraints_kernel", shape=[self.total_constraints, 1],
                                                            initializer=xavier_init())

    def _skip_constraints(self, res, d_hidden4_sigmoid, reuse):
        with tf.variable_scope("output"):
            d_out = tf.add(tf.matmul(d_hidden4_sigmoid, self.d_out_kernel), self.d_out_bias,
                           name="d_out_{}".format(reuse))
            self._log_shape("out", d_out.shape, reuse)

            return d_out

    def _apply_constraints(self, res, d_hidden4_sigmoid, constraints_placeholder, reuse):
        with tf.variable_scope("constrained_out"):
            self._log_shape("d_constraints_kernel", self.d_constraints_kernel.shape, reuse)
            input_concat = tf.concat([d_hidden4_sigmoid, constraints_placeholder],
                                     axis=1, name="input_concat_{}".format(reuse))
            self._log_shape("input_concat", input_concat.shape, reuse)
            weight_concat = tf.concat([self.d_out_kernel, self.d_constraints_kernel],
                                      axis=0, name="weight_concat_{}".format(reuse))
            self._log_shape("weight_concat", weight_concat.shape, reuse)
            d_constrained_out = tf.add(tf.matmul(input_concat, weight_concat), self.d_out_bias,
                                       name="d_constrained_out_{}".format(reuse))
            self._log_shape("constrained_out", d_constrained_out.shape, reuse)

            res["D_out_kernel"] = self.d_out_kernel
            res["D_constraints_kernel"] = self.d_constraints_kernel
            return d_constrained_out

    def _network(self, node, constraints_placeholder, reuse):
        with tf.variable_scope("discriminator", reuse=reuse):
            reuse = bool(tf.AUTO_REUSE)
            self._log_shape("in", node.shape, reuse)

            with tf.variable_scope("hidden1"):
                d_hidden1 = self.conv2d_1(node)
                self._log_shape("dh1", d_hidden1.shape, reuse)

            with tf.variable_scope("hidden2"):
                d_hidden2 = self.conv2d_2(d_hidden1)
                self._log_shape("dh2", d_hidden2.shape, reuse)
                d_hidden2 = tf.reshape(d_hidden2, [-1, np.prod(d_hidden2.shape[1:], dtype=int)])
                self._log_shape("dh2", d_hidden2.shape, reuse)

            with tf.variable_scope("hidden3"):
                d_hidden3 = self.dense_3(d_hidden2)
                self._log_shape("dh3", d_hidden3.shape, reuse)

            with tf.variable_scope("hidden4"):
                d_hidden4 = self.dense_4(d_hidden3)
                self._log_shape("dh4", d_hidden4.shape, reuse)

            with tf.variable_scope("hidden4_sigmoid"):
                d_hidden4_sigmoid = tf.nn.sigmoid(d_hidden4)
                self._log_shape("dh4_sigmoid", d_hidden4_sigmoid.shape, reuse)

            self._log_shape("d_out_kernel", self.d_out_kernel.shape, reuse)
            self._log_shape("d_out_bias", self.d_out_bias.shape, reuse)

            res = dict()
            d_out = tf.cond(self.use_constraints,
                            false_fn=lambda: self._skip_constraints(res, d_hidden4_sigmoid, reuse),
                            true_fn=lambda: self._apply_constraints(res, d_hidden4_sigmoid,
                                                                    constraints_placeholder, reuse))
            res["D_out"] = d_out
            res["D_hidden4"] = d_hidden4
            res["D_hidden4_sigmoid"] = d_hidden4_sigmoid
            return res


class Can20Discriminator64LayerAuto(CanPolygonsFloorDiscriminator):

    def __init__(self, experiment):
        super().__init__(experiment)

        with tf.variable_scope("discriminator"):
            with tf.variable_scope("hidden1"):
                self.conv2d_1 = Conv2D(filters=self.h_dim, kernel_size=5, strides=2, padding="same",
                                       activation=self.leaky_relu, kernel_initializer=xavier_init())
            with tf.variable_scope("hidden2"):
                self.conv2d_2 = Conv2D(filters=self.h_dim * 2, kernel_size=5, strides=2, padding="same",
                                       activation=self.leaky_relu, kernel_initializer=xavier_init())
            with tf.variable_scope("hidden3"):
                self.dense_3 = Dense(1024, activation=self.leaky_relu, kernel_initializer=xavier_init())
            with tf.variable_scope("hidden4"):
                self.dense_4 = Dense(64, activation=self.leaky_relu, kernel_initializer=xavier_init())
            with tf.variable_scope("shared_weights"):
                self.d_out_kernel = tf.get_variable("d_out_kernel", shape=[64, 1], initializer=xavier_init())
                self.d_out_bias = tf.get_variable("d_out_bias", shape=[1, 1], initializer=xavier_init())
            with tf.variable_scope("constrained_out"):
                self.d_constraints_kernel = tf.get_variable("d_constraints_kernel",
                                                            shape=[self.total_constraints, 1],
                                                            initializer=xavier_init())

    def _skip_constraints(self, res, d_hidden4, reuse):
        with tf.variable_scope("output"):
            d_out = tf.add(tf.matmul(d_hidden4, self.d_out_kernel), self.d_out_bias, name="d_out_{}".format(reuse))
            self._log_shape("out", d_out.shape, reuse)

            return d_out

    def _apply_constraints(self, res, d_hidden4, constraints_placeholder, reuse):
        with tf.variable_scope("constrained_out"):
            self._log_shape("d_constraints_kernel", self.d_constraints_kernel.shape, reuse)
            input_concat = tf.concat([d_hidden4, constraints_placeholder], axis=1, name="input_concat_{}".format(reuse))
            self._log_shape("input_concat", input_concat.shape, reuse)
            weight_concat = tf.concat([self.d_out_kernel, self.d_constraints_kernel],
                                      axis=0, name="weight_concat_{}".format(reuse))
            self._log_shape("weight_concat", weight_concat.shape, reuse)
            d_constrained_out = tf.add(tf.matmul(input_concat, weight_concat), self.d_out_bias,
                                       name="d_constrained_out_{}".format(reuse))
            self._log_shape("constrained_out", d_constrained_out.shape, reuse)

            return d_constrained_out

    def _network(self, node, constraints_placeholder, reuse):
        with tf.variable_scope("discriminator", reuse=reuse):
            reuse = bool(tf.AUTO_REUSE)
            self._log_shape("in", node.shape, reuse)

            with tf.variable_scope("hidden1"):
                d_hidden1 = self.conv2d_1(node)
                self._log_shape("dh1", d_hidden1.shape, reuse)

            with tf.variable_scope("hidden2"):
                d_hidden2 = self.conv2d_2(d_hidden1)
                self._log_shape("dh2", d_hidden2.shape, reuse)
                d_hidden2 = tf.reshape(d_hidden2, [-1, np.prod(d_hidden2.shape[1:], dtype=int)])
                self._log_shape("dh2", d_hidden2.shape, reuse)

            with tf.variable_scope("hidden3"):
                d_hidden3 = self.dense_3(d_hidden2)
                self._log_shape("dh3", d_hidden3.shape, reuse)

            with tf.variable_scope("hidden4"):
                d_hidden4 = self.dense_4(d_hidden3)
                self._log_shape("dh4", d_hidden4.shape, reuse)

            self._log_shape("d_out_kernel", self.d_out_kernel.shape, reuse)
            self._log_shape("d_out_bias", self.d_out_bias.shape, reuse)

            res = dict()
            d_out = tf.cond(self.use_constraints,
                            false_fn=lambda: self._skip_constraints(res, d_hidden4, reuse),
                            true_fn=lambda: self._apply_constraints(res, d_hidden4, constraints_placeholder, reuse))

            res["D_out"] = d_out
            res["D_hidden4"] = d_hidden4
            return res


class Can20DiscriminatorLastLayerAuto(CanPolygonsFloorDiscriminator):

    def __init__(self, experiment):
        super().__init__(experiment)

        with tf.variable_scope("discriminator"):
            with tf.variable_scope("hidden1"):
                self.conv2d_1 = Conv2D(filters=self.h_dim, kernel_size=5, strides=2, padding="same",
                                       activation=self.leaky_relu, kernel_initializer=xavier_init())
            with tf.variable_scope("hidden2"):
                self.conv2d_2 = Conv2D(filters=self.h_dim * 2, kernel_size=5, strides=2, padding="same",
                                       activation=self.leaky_relu, kernel_initializer=xavier_init())
            with tf.variable_scope("hidden3"):
                self.dense_3 = Dense(1024, activation=self.leaky_relu, kernel_initializer=xavier_init())
            with tf.variable_scope("output"):
                self.dense_output = Dense(1, activation=self.leaky_relu, kernel_initializer=xavier_init())
            with tf.variable_scope("constrained_ll_out"):
                self.dense_constrained_ll_out = Dense(1, activation=self.leaky_relu, kernel_initializer=xavier_init())
                self.d_out_kernel = tf.get_variable("d_out_kernel", shape=[2, 1], initializer=xavier_init())
                self.d_out_bias = tf.get_variable("d_out_bias", shape=[1, 1], initializer=xavier_init())

    def _skip_constraints(self, res, d_out):
        return d_out

    def _apply_constraints(self, res, d_out, constraints_placeholder, reuse):
        with tf.variable_scope("constrained_ll_out"):
            constraints_out = self.dense_constrained_ll_out(constraints_placeholder)
            self._log_shape("d_out_kernel", self.d_out_kernel.shape, reuse)
            self._log_shape("d_out_bias", self.d_out_bias.shape, reuse)

            input_concat = tf.concat([d_out, constraints_out], axis=1, name="input_concat_{}".format(reuse))
            self._log_shape("input_concat", input_concat.shape, reuse)
            d_constrained_out = tf.add(tf.matmul(input_concat, self.d_out_kernel), self.d_out_bias,
                                       name="d_constrained_out_{}".format(reuse))
            self._log_shape("constrained_out", d_constrained_out.shape, reuse)

            return d_constrained_out

    def _network(self, node, constraints_placeholder, reuse):
        with tf.variable_scope("discriminator", reuse=reuse):
            reuse = bool(tf.AUTO_REUSE)
            self._log_shape("in", node.shape, reuse)

            with tf.variable_scope("hidden1"):
                d_hidden1 = self.conv2d_1(node)
                self._log_shape("dh1", d_hidden1.shape, reuse)
            with tf.variable_scope("hidden2"):
                d_hidden2 = self.conv2d_2(d_hidden1)
                self._log_shape("dh2", d_hidden2.shape, reuse)
                d_hidden2 = tf.reshape(d_hidden2, [-1, np.prod(d_hidden2.shape[1:], dtype=int)])
                self._log_shape("dh2", d_hidden2.shape, reuse)
            with tf.variable_scope("hidden3"):
                d_hidden3 = self.dense_3(d_hidden2)
                self._log_shape("dh3", d_hidden3.shape, reuse)
            with tf.variable_scope("output"):
                d_out = self.dense_output(d_hidden3)
                self._log_shape("out", d_out.shape, reuse)

            res = dict()
            d_out = tf.cond(self.use_constraints,
                            false_fn=lambda: self._skip_constraints(res, d_out),
                            true_fn=lambda: self._apply_constraints(res, d_out, constraints_placeholder, reuse))

            res["D_out"] = d_out
            return res


# d_out, d_constrained_out
class Can60Discriminator32LayerAuto(CanPolygonsFloorDiscriminator):

    def __init__(self, experiment):
        super().__init__(experiment)

        with tf.variable_scope("discriminator"):
            with tf.variable_scope("hidden1"):
                self.conv2d_1 = Conv2D(filters=self.h_dim, kernel_size=5, strides=2, padding="same",
                                       activation=self.leaky_relu, kernel_initializer=xavier_init())
            with tf.variable_scope("hidden15"):
                self.conv2d_15 = Conv2D(filters=self.h_dim, kernel_size=5, strides=2,
                                        padding="same", activation=self.leaky_relu,
                                        kernel_initializer=xavier_init())
            with tf.variable_scope("hidden2"):
                self.conv2d_2 = Conv2D(filters=self.h_dim * 2, kernel_size=5, strides=2, padding="same",
                                       activation=self.leaky_relu, kernel_initializer=xavier_init())
            with tf.variable_scope("hidden3"):
                self.dense_3 = Dense(1024, activation=self.leaky_relu, kernel_initializer=xavier_init())
            with tf.variable_scope("hidden4"):
                self.dense_4 = Dense(32, activation=self.leaky_relu, kernel_initializer=xavier_init())
            with tf.variable_scope("shared_weights"):
                self.d_out_kernel = tf.get_variable("d_out_kernel", shape=[32, 1], initializer=xavier_init())
                self.d_out_bias = tf.get_variable("d_out_bias", shape=[1, 1], initializer=xavier_init())
            with tf.variable_scope("constrained_out"):
                self.d_constraints_kernel = tf.get_variable("d_constraints_kernel", shape=[self.total_constraints, 1],
                                                            initializer=xavier_init())

    def _skip_constraints(self, res, d_hidden4, reuse):
        with tf.variable_scope("output"):
            d_out = tf.add(tf.matmul(d_hidden4, self.d_out_kernel), self.d_out_bias, name="d_out_{}".format(reuse))
            self._log_shape("out", d_out.shape, reuse)

            return d_out

    def _apply_constraints(self, res, d_hidden4, constraints_placeholder, reuse):
        with tf.variable_scope("constrained_out"):
            self._log_shape("d_constraints_kernel", self.d_constraints_kernel.shape, reuse)
            input_concat = tf.concat([d_hidden4, constraints_placeholder], axis=1, name="input_concat_{}".format(reuse))
            self._log_shape("input_concat", input_concat.shape, reuse)
            weight_concat = tf.concat([self.d_out_kernel, self.d_constraints_kernel],
                                      axis=0, name="weight_concat_{}".format(reuse))
            self._log_shape("weight_concat", weight_concat.shape, reuse)
            d_constrained_out = tf.add(tf.matmul(input_concat, weight_concat), self.d_out_bias,
                                       name="d_constrained_out_{}".format(reuse))
            self._log_shape("constrained_out", d_constrained_out.shape, reuse)

            return d_constrained_out

    def _network(self, node, constraints_placeholder, reuse):
        with tf.variable_scope("discriminator", reuse=reuse):
            reuse = bool(tf.AUTO_REUSE)
            self._log_shape("in", node.shape, reuse)

            with tf.variable_scope("hidden1"):
                d_hidden1 = self.conv2d_1(node)
                self._log_shape("dh1", d_hidden1.shape, reuse)

            with tf.variable_scope("hidden15"):
                d_hidden15 = self.conv2d_15(d_hidden1)
                self._log_shape("dh15", d_hidden15.shape, reuse)

            with tf.variable_scope("hidden2"):
                d_hidden2 = self.conv2d_2(d_hidden15)
                self._log_shape("dh2", d_hidden2.shape, reuse)
                d_hidden2 = tf.reshape(d_hidden2, [-1, np.prod(d_hidden2.shape[1:], dtype=int)])
                self._log_shape("dh2", d_hidden2.shape, reuse)

            with tf.variable_scope("hidden3"):
                d_hidden3 = self.dense_3(d_hidden2)
                self._log_shape("dh3", d_hidden3.shape, reuse)

            with tf.variable_scope("hidden4"):
                d_hidden4 = self.dense_4(d_hidden3)
                self._log_shape("dh4", d_hidden4.shape, reuse)

            self._log_shape("d_out_kernel", self.d_out_kernel.shape, reuse)
            self._log_shape("d_out_bias", self.d_out_bias.shape, reuse)

            res = dict()

            d_out = tf.cond(self.use_constraints,
                            false_fn=lambda: self._skip_constraints(res, d_hidden4, reuse),
                            true_fn=lambda: self._apply_constraints(res, d_hidden4, constraints_placeholder, reuse))

            res["D_out"] = d_out
            res["D_hidden4"] = d_hidden4
            return res
