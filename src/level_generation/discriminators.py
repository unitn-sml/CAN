"""
File containing the level discriminator architectures.
"""
import tensorflow as tf
from base_layers import Discriminator

from tensorflow.initializers import random_normal
from tensorflow import orthogonal_initializer as ort_init, zeros_initializer as zeros_init, expand_dims
from tensorflow.keras.layers import Layer, Dense, BatchNormalization, Conv2D, ReLU, LeakyReLU
from tensorflow.contrib.distributions import OneHotCategorical


class DCGAN_D(Discriminator):

    def __init__(self, experiment):
        super().__init__(experiment)

        self.logger = self.experiment["LOGGER"]
        self.shape = self.experiment['SHAPE']
        self.batch_size = self.experiment["BATCH_SIZE"]
        self.bgan_samples = self.experiment["NUM_BGAN_SAMPLES"]
        self.isize = self.experiment['ISIZE']
        self.number_filters = self.experiment['NUMBER_FILTERS_DISCRIMINATOR']
        self.leakiness = self.experiment['LEAKINESS']
        self.clamp_high = self.experiment["CLAMP_HIGH"]
        self.clamp_low = self.experiment["CLAMP_LOW"]
        self.normal_initializer = random_normal(mean=0.0, stddev=0.02)
        self.affine_normal_initializer = random_normal(mean=1.0, stddev=0.02)
        self.weights_constraints = lambda x: tf.clip_by_value(x, self.clamp_low, self.clamp_high)
        self.zero_initializer = tf.zeros_initializer()
        # placeholder that will decide to use or not constrained training
        self.use_constraints = tf.placeholder(tf.bool, shape=())

        assert self.isize % 16 == 0, "isize has to be a multiple of 16"

        self.batch_norm_args = {
            'axis': -1,
            'momentum': 0.9,
            'epsilon': 1e-5,
            'gamma_initializer': self.affine_normal_initializer,
            'beta_initializer': self.zero_initializer,
            'gamma_constraint': self.weights_constraints,
            'beta_constraint': self.weights_constraints
        }

        with tf.variable_scope("discriminator"):
            with tf.variable_scope("conv_initial"):
                self.conv_1 = Conv2D(filters=self.number_filters,
                                     kernel_size=(4, 4),
                                     strides=(2, 2),
                                     padding='SAME',
                                     use_bias=False,
                                     kernel_initializer=self.normal_initializer,
                                     kernel_constraint=self.weights_constraints)
                self.leaky_relu_1 = LeakyReLU(alpha=self.leakiness)

            with tf.variable_scope("conv_iter"):
                csize, cndf = self.isize / 2, self.number_filters
                self.layers_list = []
                while csize > 4:
                    in_feat = cndf
                    out_feat = cndf * 2
                    self.layers_list.append(Conv2D(filters=out_feat,
                                                   kernel_size=(4, 4),
                                                   strides=(2, 2),
                                                   padding='SAME',
                                                   use_bias=False,
                                                   kernel_initializer=self.normal_initializer,
                                                   kernel_constraint=self.weights_constraints))
                    self.layers_list.append(BatchNormalization(**self.batch_norm_args))
                    self.layers_list.append(LeakyReLU(alpha=self.leakiness))
                    cndf = cndf * 2
                    csize = csize / 2

            with tf.variable_scope("conv_final"):
                self.conv_final = Conv2D(filters=1,
                                         kernel_size=(4, 4),
                                         strides=(1, 1),
                                         padding='VALID',
                                         use_bias=False,
                                         kernel_initializer=self.normal_initializer,
                                         kernel_constraint=self.weights_constraints)

    def _pre_processing(self, **graph_nodes):
        self._requires("G_probs", **graph_nodes)

    def _forward(self, **graph_nodes):
        D_real = self._network(self.X, False)
        G_probs = graph_nodes["G_probs"]

        # from (batch, num bgan samples, shape) to (-1, shape)
        D_fake = self._network(G_probs, True)

        nodes = dict()
        nodes["D_real"] = D_real
        nodes["D_fake"] = D_fake
        nodes["use_constraints"] = self.use_constraints
        nodes["X"] = self.X

        return nodes, dict(), dict()

    def _network(self, node, reuse):
        msg = "D_SHAPE {}: {}"
        self.logger.debug(msg.format("in", node.shape))

        # padding input samples to 32x32
        paddings = tf.constant([[0, 0], [0, 32-int(node.shape[1])], [0, 32-int(node.shape[2])], [0, 0]])
        node = tf.pad(node, paddings, "CONSTANT", constant_values=0)

        # input with shape (batch_size, z_dim)
        with tf.variable_scope("discriminator", reuse=reuse):
            with tf.variable_scope("conv_initial"):
                x = self.conv_1(node)
                self.logger.debug(msg.format("conv_1", x.shape))
                x = self.leaky_relu_1(x)
                self.logger.debug(msg.format("relu_1", x.shape))

            with tf.variable_scope("conv_iter"):
                for layer in self.layers_list:
                    if layer.__class__.__name__ == 'BatchNormalizationV1':
                        x = layer(x, training=True)
                    else:
                        x = layer(x)
                    self.logger.debug(msg.format(layer.__class__.__name__, x.shape))

            with tf.variable_scope("conv_final"):
                d_out = self.conv_final(x)
                self.logger.debug(msg.format("conv_final", d_out.shape))

        return d_out
