import numpy as np
import tensorflow as tf

from tensorflow import (orthogonal_initializer as ort_init, zeros_initializer as zeros_init)
from tensorflow.python.ops.nn import relu

from tensorflow.keras.layers import Layer, Dense, BatchNormalization, Conv2DTranspose
from tensorflow.contrib.layers import xavier_initializer as xavier_init
from polygons_floor.discriminators import CanPolygonsFloorDiscriminator


def spectral_norm(w, iteration=1):
    """
    See https://openreview.net/pdf?id=B1QRgziT-
    Implementation from: https://github.com/taki0112/Spectral_Normalization-Tensorflow

    :param w:
    :param iteration:
    :return:
    """

    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
       power iteration
       Usually iteration = 1 will be enough
       """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm


def conv(x, channels, kernel=4, stride=2, padding="VALID", use_bias=True, sn=False, scope='conv_0'):
    """
    Slightly altered implementation from: https://github.com/taki0112/Spectral_Normalization-Tensorflow

    :param x:
    :param channels:
    :param kernel:
    :param stride:
    :param padding:
    :param use_bias:
    :param sn:
    :param scope:
    :return:
    """
    with tf.variable_scope(scope):

        if sn:
            w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels],
                                initializer=xavier_init())
            bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
            x = tf.nn.conv2d(input=x, filter=spectral_norm(w), strides=[1, stride, stride, 1], padding=padding)
            if use_bias:
                x = tf.nn.bias_add(x, bias)

        else:
            x = tf.layers.conv2d(inputs=x, filters=channels,
                                 kernel_size=kernel, kernel_initializer=xavier_init(),
                                 strides=stride, use_bias=use_bias, padding=padding)

        return x


def fully_conneted(x, units, use_bias=True, sn=False, scope='fully_0'):
    """
    Implementation from: https://github.com/taki0112/Spectral_Normalization-Tensorflow

    :param x:
    :param units:
    :param use_bias:
    :param sn:
    :param scope:
    :return:
    """
    with tf.variable_scope(scope):
        x = tf.layers.flatten(x)
        shape = x.get_shape().as_list()
        channels = shape[-1]

        if sn:
            w = tf.get_variable("kernel", [channels, units], tf.float32,
                                initializer=xavier_init())
            if use_bias:
                bias = tf.get_variable("bias", [units],
                                       initializer=tf.constant_initializer(0.0))

                x = tf.matmul(x, spectral_norm(w)) + bias
            else:
                x = tf.matmul(x, spectral_norm(w))

        else:
            x = tf.layers.dense(x, units=units, kernel_initializer=xavier_init(), use_bias=use_bias)

        return x


class SnDiscriminator(CanPolygonsFloorDiscriminator):
    """
    A discriminator making use of spectral normalization.

    """

    def __init__(self, experiment):
        super().__init__(experiment)

    def _network(self, node, constraints_placeholder, reuse):
        with tf.variable_scope("discriminator", reuse=reuse):
            reuse = bool(tf.AUTO_REUSE)
            self._log_shape("in", node.shape, reuse)

            d_hidden1 = conv(node, self.h_dim, kernel=5, stride=2, padding="SAME", use_bias=True, sn=True,
                             scope="hidden1")
            d_hidden1 = self.leaky_relu(d_hidden1)
            self._log_shape("dh1", d_hidden1.shape, reuse)

            d_hidden2 = conv(d_hidden1, self.h_dim * 2, kernel=5, stride=2, padding="SAME", use_bias=True, sn=True,
                             scope="hidden2")
            self._log_shape("dh2", d_hidden1.shape, reuse)
            d_hidden2 = tf.reshape(d_hidden2, [-1, np.prod(d_hidden2.shape[1:], dtype=int)])
            d_hidden2 = self.leaky_relu(d_hidden2)
            self._log_shape("dh2", d_hidden2.shape, reuse)

            d_hidden3 = fully_conneted(d_hidden2, units=1024, use_bias=True, sn=True, scope="hidden3")
            d_hidden3 = self.leaky_relu(d_hidden3)
            self._log_shape("dh3", d_hidden3.shape, reuse)

            d_hidden4 = fully_conneted(d_hidden3, units=32, use_bias=True, sn=True, scope="hidden4")
            d_hidden4 = self.leaky_relu(d_hidden4)
            self._log_shape("dh4", d_hidden4.shape, reuse)

            d_out = fully_conneted(d_hidden4, units=1, use_bias=True, sn=True, scope="d_out")
            self._log_shape("out", d_out.shape, reuse)

            res = dict()

            res["D_out"] = d_out
            return res


class Sn28Discriminator(CanPolygonsFloorDiscriminator):
    """
    A discriminator making use of spectral normalization.

    """

    def __init__(self, experiment):
        super().__init__(experiment)

    def _network(self, node, constraints_placeholder, reuse):
        with tf.variable_scope("discriminator", reuse=reuse):
            reuse = bool(tf.AUTO_REUSE)
            self._log_shape("in", node.shape, reuse)

            d_hidden1 = conv(node, self.h_dim * 4, kernel=4, stride=1, padding="SAME", use_bias=True, sn=True,
                             scope="hidden1")
            d_hidden1 = self.leaky_relu(d_hidden1)
            self._log_shape("dh1", d_hidden1.shape, reuse)

            d_hidden2 = conv(d_hidden1, self.h_dim, kernel=4, stride=2, padding="SAME", use_bias=True, sn=True,
                             scope="hidden2")
            self._log_shape("dh2", d_hidden1.shape, reuse)
            d_hidden2 = tf.reshape(d_hidden2, [-1, np.prod(d_hidden2.shape[1:], dtype=int)])
            d_hidden2 = self.leaky_relu(d_hidden2)
            self._log_shape("dh2", d_hidden2.shape, reuse)

            d_hidden3 = fully_conneted(d_hidden2, units=self.h_dim * 7 * 7, use_bias=True, sn=True, scope="hidden3")
            d_hidden3 = self.leaky_relu(d_hidden3)
            self._log_shape("dh3", d_hidden3.shape, reuse)

            d_hidden4 = fully_conneted(d_hidden3, units=self.h_dim * 16, use_bias=True, sn=True, scope="hidden4")
            d_hidden4 = self.leaky_relu(d_hidden4)
            self._log_shape("dh4", d_hidden4.shape, reuse)

            d_out = fully_conneted(d_hidden4, units=1, use_bias=True, sn=True, scope="d_out")
            self._log_shape("out", d_out.shape, reuse)

            res = dict()

            res["D_out"] = d_out
            return res


class GanFormulaDiscriminator(CanPolygonsFloorDiscriminator):

    def __init__(self, experiment):
        super().__init__(experiment)

        with tf.variable_scope("discriminator"):
            with tf.variable_scope("hidden1"):
                self.dense_1 = Dense(self.h_dim * 4, activation=relu)
                self.batch_normalization_1 = BatchNormalization()
            with tf.variable_scope("hidden2"):
                self.dense_2 = Dense(self.h_dim * 2, activation=relu)
                self.batch_normalization_2 = BatchNormalization()
            with tf.variable_scope("hidden3"):
                self.dense_3 = Dense(self.h_dim, activation=relu)
                self.batch_normalization_3 = BatchNormalization()
            with tf.variable_scope("output"):
                self.dense_output = Dense(1, kernel_initializer=xavier_init())

    def _network(self, node, constraints_placeholder, reuse):
        msg = "D_SHAPE {} {} [reuse={}]"

        with tf.variable_scope("discriminator", reuse=reuse):
            reuse = bool(tf.AUTO_REUSE)
            self.logger.debug(msg.format("in", node.shape, reuse))

            with tf.variable_scope("hidden1"):
                d_hidden1 = self.dense_1(node)
                self.logger.debug(msg.format("dh1", d_hidden1.shape, reuse))

            with tf.variable_scope("hidden2"):
                d_hidden2 = self.dense_2(d_hidden1)
                self.logger.debug(msg.format("dh2", d_hidden2.shape, reuse))

            with tf.variable_scope("hidden3"):
                d_hidden3 = self.dense_3(d_hidden2)
                self.logger.debug(msg.format("dh3", d_hidden3.shape, reuse))

            with tf.variable_scope("output"):
                d_out = self.dense_output(d_hidden3)
                self.logger.debug(msg.format("out", d_out.shape, reuse))

        res = dict()

        res["D_out"] = d_out
        return res