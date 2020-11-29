"""
File containing the common generators architectures.
"""
import tensorflow as tf
from base_layers import Generator

from tensorflow import (orthogonal_initializer as ort_init, zeros_initializer as zeros_init)
from tensorflow.python.ops.nn import relu
from utils.utils_tf import pad_left

from tensorflow.keras.layers import Layer, Dense, BatchNormalization, Conv2DTranspose

# some architectures (and hyper-parameters) are drawn from the original work on
# "Boundary-Seeking Generative Adversarial Networks"
# ref: https://github.com/rdevon/BGAN
# The original code was written in Theano/Lasagne and has been ported to TF

# NOTE: conv2d/deconv2d paddings may be slightly different from the ones in
# Lasagne. To make them equivalent, a manual pad2 is necessary. Sketch:
# pad2 = lambda x: tf.pad(x, [[0, 0], [2, 2], [2, 2], [0, 0]], "CONSTANT")

# NOTE: this batch normalization is slightly different from the one in Lasagne.
# See the LS-TF porting notes for further info.
from functools import partial


class PolygonsFloorGenerator(Generator):

    def __init__(self, experiment):
        super().__init__(experiment)
        self.h_dim = experiment["H_DIM"]
        self.logger = experiment["LOGGER"]
        self.colors = experiment["SHAPE"][-1]
        self.training = True

    def _pre_processing(self, **graph_nodes):
        pass

    def binomial_sampling(self, G_output_logit):
        """
        Performs the binomial sampling step, given the output logit, it will perform a sigmoid operation
        and a sampling operation based on uniform noise. Note that this will also add a placeholder, R,
        of shape (bgan samples, bs, shape) which
        is used to perform the uniform random sampling, it must be filled with random numbers [0,1), you can (should)
        use the Computable named RandomUniform which is provided.
        TTheoretically you should not need this placeholder, but it is needed to make sure
        that constrained training is correctly run by producing the same exact
        samples which are used to compute the constraints results before passing them to the
        discriminator.
        :param G_output_logit: Output node of the network, pre activation.
        :return: The G_sample node that will contain/produce discrete binomial samples from the network (0/1).
        """

        G_output_logit = tf.sigmoid(G_output_logit)
        # G_output will be replicated NUM_BGAN_SAMPLES times to be legally compared with R, the uniform noise
        # G_sample will be the real batch of images with black or white pixels
        G_sample = tf.cast(self.R <= pad_left(G_output_logit), tf.float32)
        return G_sample

    def multinomial_sampling(self, G_output_logit):
        """
        Performs the multinomial sampling step, given the output logit, it will perform a softmax operation
        and a sampling operation based on noise.
        :param G_output_logit: Output node of the network, pre activation.
        :return: The G_sample node that will contain/produce discrete multinomial samples from the network (~ 1 hot).
        """
        bs = self.experiment["BATCH_SIZE"]
        shape = self.experiment["SHAPE"]
        bgan_samples = self.experiment["NUM_BGAN_SAMPLES"]

        G_output_logit = tf.reshape(G_output_logit, (-1, shape[-1]))
        G_output_logit = tf.nn.softmax(G_output_logit)
        G_output_logit = tf.reshape(G_output_logit, (bs, shape[0], shape[1], shape[2]))
        p_t = tf.tile(pad_left(G_output_logit), (bgan_samples, 1, 1, 1, 1))
        p = tf.reshape(p_t, (-1, shape[-1]))

        multinomial_generator = tf.contrib.distributions.OneHotCategorical(probs=p, dtype=tf.float32)
        G_sample = multinomial_generator.sample()
        G_sample = tf.reshape(G_sample, [bgan_samples, bs] + shape)
        return G_sample


class GanGenerator(PolygonsFloorGenerator):

    def __init__(self, experiment):
        super().__init__(experiment)
        self.R = tf.placeholder(tf.float32, [self.experiment["NUM_BGAN_SAMPLES"],
                                             self.experiment["BATCH_SIZE"]] + self.experiment["SHAPE"], "R_placeholder")


class Gan16Generator(GanGenerator):

    def __init__(self, experiment):
        super().__init__(experiment)

        with tf.variable_scope("generator"):
            with tf.variable_scope("hidden1"):
                self.dense_1 = Dense(1024, activation=relu)
                self.batch_normalization_1 = BatchNormalization()
            with tf.variable_scope("hidden2"):
                self.dense_2 = Dense(self.h_dim * 2 * 4 * 4, activation=relu)
                self.batch_normalization_2 = BatchNormalization()
            with tf.variable_scope("hidden3"):
                self.conv2d_transpose_3 = Conv2DTranspose(filters=self.h_dim, kernel_size=5, strides=2, padding="same",
                                                          activation=relu, kernel_initializer=ort_init(),
                                                          bias_initializer=zeros_init())
                self.batch_normalization_3 = BatchNormalization()
            with tf.variable_scope("output"):
                self.conv2d_transpose_output = Conv2DTranspose(filters=self.colors, kernel_size=5, strides=2,
                                                               padding="same", kernel_initializer=ort_init(),
                                                               bias_initializer=zeros_init())

    def _forward(self, **graph_nodes):
        msg = "G_SHAPE {}: {}"
        self.logger.debug(msg.format("in", self.z.shape))

        with tf.variable_scope("generator"):
            with tf.variable_scope("hidden1"):
                g_hidden1 = self.batch_normalization_1(self.dense_1(self.z), self.training)
                self.logger.debug(msg.format("gh1", g_hidden1.shape))

            with tf.variable_scope("hidden2"):
                g_hidden2 = self.batch_normalization_2(self.dense_2(g_hidden1), self.training)
                self.logger.debug(msg.format("gh2", g_hidden2.shape))
                g_hidden2 = tf.reshape(g_hidden2, [-1, self.h_dim * 2, 4, 4])
                self.logger.debug(msg.format("gh2", g_hidden2.shape))
                # deconv2d only supports nhwc channels. Transposing nchw to nhwc
                g_hidden2 = tf.transpose(g_hidden2, [0, 3, 2, 1])
                self.logger.debug(msg.format("gh3", g_hidden2.shape))

            with tf.variable_scope("hidden3"):
                g_hidden3 = self.batch_normalization_3(self.conv2d_transpose_3(g_hidden2), self.training)
                self.logger.debug(msg.format("gh3", g_hidden3.shape))

            with tf.variable_scope("output"):
                g_out = self.conv2d_transpose_output(g_hidden3)
                self.logger.debug(msg.format("out", g_out.shape))

                # if experiment is binomial
                if self.colors == 1:
                    G_sample = self.binomial_sampling(g_out)
                else:
                    G_sample = self.multinomial_sampling(g_out)

        res = dict()
        res["generator"] = self
        res["z"] = self.z
        res["R"] = self.R
        res["G_output_logit"] = g_out
        res["G_sample"] = G_sample

        return res, dict(), dict()


class Gan20Generator(GanGenerator):

    def __init__(self, experiment):
        super().__init__(experiment)

        with tf.variable_scope("generator"):
            with tf.variable_scope("hidden1"):
                self.dense_1 = Dense(1024, activation=relu)
                self.batch_normalization_1 = BatchNormalization()
            with tf.variable_scope("hidden2"):
                self.dense_2 = Dense(self.h_dim * 2 * 5 * 5, activation=relu)
                self.batch_normalization_2 = BatchNormalization()
            with tf.variable_scope("hidden3"):
                self.conv2d_transpose_3 = Conv2DTranspose(filters=self.h_dim, kernel_size=5, strides=2, padding="same",
                                                          activation=relu, kernel_initializer=ort_init(),
                                                          bias_initializer=zeros_init())
                self.batch_normalization_3 = BatchNormalization()
            with tf.variable_scope("output"):
                self.conv2d_transpose_output = Conv2DTranspose(filters=self.colors, kernel_size=5, strides=2,
                                                               padding="same", kernel_initializer=ort_init(),
                                                               bias_initializer=zeros_init())

    def _forward(self, **graph_nodes):
        msg = "G_SHAPE {}: {}"
        self.logger.debug(msg.format("in", self.z.shape))

        with tf.variable_scope("generator"):
            with tf.variable_scope("hidden1"):
                g_hidden1 = self.batch_normalization_1(self.dense_1(self.z), self.training)
                self.logger.debug(msg.format("gh1", g_hidden1.shape))

            with tf.variable_scope("hidden2"):
                g_hidden2 = self.batch_normalization_2(self.dense_2(g_hidden1), self.training)
                self.logger.debug(msg.format("gh2", g_hidden2.shape))
                g_hidden2 = tf.reshape(g_hidden2, [-1, self.h_dim * 2, 5, 5])
                self.logger.debug(msg.format("gh2", g_hidden2.shape))

            with tf.variable_scope("hidden3"):
                # deconv2d only supports nhwc channels. Transposing nchw to nhwc
                g_hidden2 = tf.transpose(g_hidden2, [0, 3, 2, 1])
                self.logger.debug(msg.format("gh3", g_hidden2.shape))
                g_hidden3 = self.batch_normalization_3(self.conv2d_transpose_3(g_hidden2), self.training)
                self.logger.debug(msg.format("gh3", g_hidden3.shape))

            with tf.variable_scope("output"):
                g_out = self.conv2d_transpose_output(g_hidden3)
                self.logger.debug(msg.format("out", g_out.shape))

                # if experiment is binomial
                if self.colors == 1:
                    G_sample = self.binomial_sampling(g_out)
                else:
                    G_sample = self.multinomial_sampling(g_out)

        res = dict()
        res["generator"] = self
        res["z"] = self.z
        res["R"] = self.R
        res["G_output_logit"] = g_out
        res["G_sample"] = G_sample

        return res, dict(), dict()


class Gan28Generator(GanGenerator):

    def __init__(self, experiment):
        super().__init__(experiment)

        with tf.variable_scope("generator"):
            with tf.variable_scope("hidden1"):
                self.dense_1 = Dense(self.h_dim * 8, activation=relu)
                self.batch_normalization_1 = BatchNormalization()

            with tf.variable_scope("hidden2"):
                self.dense_2 = Dense(self.h_dim * 16, activation=relu)
                self.batch_normalization_2 = BatchNormalization()

            with tf.variable_scope("hidden3"):
                self.dense_3 = Dense(self.h_dim * 7 * 7, activation=relu)
                self.batch_normalization_3 = BatchNormalization()

            with tf.variable_scope("hidden4"):
                self.conv2d_transpose_3 = Conv2DTranspose(filters=self.h_dim * 4, kernel_size=5, strides=2,
                                                          padding="same", activation=relu,
                                                          kernel_initializer=ort_init(), bias_initializer=zeros_init())
                self.batch_normalization_4 = BatchNormalization()

            with tf.variable_scope("output"):
                self.conv2d_transpose_output = Conv2DTranspose(filters=self.colors, kernel_size=3, strides=2,
                                                               padding="same", kernel_initializer=ort_init(),
                                                               bias_initializer=zeros_init())

    def _forward(self, **graph_nodes):
        msg = "G_SHAPE {}: {}"
        self.logger.debug(msg.format("in", self.z.shape))

        with tf.variable_scope("generator"):
            with tf.variable_scope("hidden1"):
                g_hidden1 = self.batch_normalization_1(self.dense_1(self.z), self.training)
                self.logger.debug(msg.format("gh1", g_hidden1.shape))

            with tf.variable_scope("hidden2"):
                g_hidden2 = self.batch_normalization_2(self.dense_2(g_hidden1), self.training)
                self.logger.debug(msg.format("gh2", g_hidden2.shape))

            with tf.variable_scope("hidden3"):
                g_hidden3 = self.batch_normalization_3(self.dense_3(g_hidden2), self.training)
                self.logger.debug(msg.format("gh3", g_hidden3.shape))
                g_hidden3 = tf.reshape(g_hidden3, [-1, self.h_dim, 7, 7])
                self.logger.debug(msg.format("gh3", g_hidden3.shape))

            with tf.variable_scope("hidden4"):
                g_hidden4 = tf.transpose(g_hidden3, [0, 3, 2, 1])
                self.logger.debug(msg.format("gh4", g_hidden4.shape))
                # deconv2d only supports nhwc channels. Transposing nchw to nhwc
                g_hidden4 = self.batch_normalization_4(self.conv2d_transpose_3(g_hidden4), self.training)
                self.logger.debug(msg.format("gh4", g_hidden4.shape))

            with tf.variable_scope("output"):
                g_out = self.conv2d_transpose_output(g_hidden4)
                self.logger.debug(msg.format("out", g_out.shape))

                # if experiment is binomial
                if self.colors == 1:
                    G_sample = self.binomial_sampling(g_out)
                else:
                    G_sample = self.multinomial_sampling(g_out)

        res = dict()
        res["generator"] = self
        res["z"] = self.z
        res["R"] = self.R
        res["G_output_logit"] = g_out
        res["G_sample"] = G_sample

        return res, dict(), dict()


class Gan28GeneratorNoActNoBias(GanGenerator):

    def __init__(self, experiment):
        super().__init__(experiment)

        with tf.variable_scope("generator"):
            with tf.variable_scope("hidden1"):
                self.dense_1 = Dense(1024, activation=None, use_bias=False)
                self.batch_normalization_1 = BatchNormalization()

            with tf.variable_scope("hidden2"):
                self.dense_2 = Dense(self.h_dim * 2 * 7 * 7, activation=None, use_bias=False)
                self.batch_normalization_2 = BatchNormalization()

            with tf.variable_scope("hidden3"):
                self.conv2d_transpose_3 = Conv2DTranspose(filters=self.h_dim, kernel_size=5, strides=2, padding="same",
                                                          kernel_initializer=ort_init(), bias_initializer=zeros_init(),
                                                          use_bias=False, activation=None)
                self.batch_normalization_3 = BatchNormalization()

            with tf.variable_scope("output"):
                self.conv2d_transpose_output = Conv2DTranspose(filters=self.colors, kernel_size=5, strides=2,
                                                               padding="same", kernel_initializer=ort_init(),
                                                               bias_initializer=zeros_init())

    def _forward(self, **graph_nodes):
        msg = "G_SHAPE {}: {}"
        self.logger.debug(msg.format("in", self.z.shape))

        with tf.variable_scope("generator"):
            with tf.variable_scope("hidden1"):
                g_hidden1 = relu(self.batch_normalization_1(self.dense_1(self.z)))
                self.logger.debug(msg.format("gh1", g_hidden1.shape))

            with tf.variable_scope("hidden2"):
                g_hidden2 = relu(self.batch_normalization_2(self.dense_2(g_hidden1)))
                self.logger.debug(msg.format("gh2", g_hidden2.shape))
                g_hidden2 = tf.reshape(g_hidden2, [-1, self.h_dim * 2, 7, 7])
                self.logger.debug(msg.format("gh2", g_hidden2.shape))

            with tf.variable_scope("hidden3"):
                # deconv2d only supports nhwc channels. Transposing nchw to nhwc
                g_hidden2 = tf.transpose(g_hidden2, [0, 3, 2, 1])
                self.logger.debug(msg.format("gh3", g_hidden2.shape))
                g_hidden3 = relu(self.batch_normalization_3(self.conv2d_transpose_3(g_hidden2)))
                self.logger.debug(msg.format("gh3", g_hidden3.shape))

            with tf.variable_scope("output"):
                g_out = self.conv2d_transpose_output(g_hidden3)
                self.logger.debug(msg.format("out", g_out.shape))

                # if experiment is binomial
                if self.colors == 1:
                    G_sample = self.binomial_sampling(g_out)
                else:
                    G_sample = self.multinomial_sampling(g_out)

        res = dict()
        res["generator"] = self
        res["z"] = self.z
        res["R"] = self.R
        res["G_output_logit"] = g_out
        res["G_sample"] = G_sample

        return res, dict(), dict()


class Gan32Generator(GanGenerator):

    def __init__(self, experiment):
        super().__init__(experiment)

        with tf.variable_scope("generator"):
            with tf.variable_scope("hidden1"):
                self.dense_1 = Dense(self.h_dim * 4 * 4 * 4, activation=relu)
                self.batch_normalization_1 = BatchNormalization()
            with tf.variable_scope("hidden2"):
                self.conv2d_transpose_2 = Conv2DTranspose(filters=self.h_dim * 2, kernel_size=5, strides=2,
                                                          padding="same", activation=relu,
                                                          kernel_initializer=ort_init(), bias_initializer=zeros_init())
                self.batch_normalization_2 = BatchNormalization()
            with tf.variable_scope("hidden3"):
                self.conv2d_transpose_3 = Conv2DTranspose(filters=self.h_dim, kernel_size=5, strides=2,
                                                          padding="same", activation=relu,
                                                          kernel_initializer=ort_init(), bias_initializer=zeros_init())
                self.batch_normalization_3 = BatchNormalization()
            with tf.variable_scope("output"):
                self.conv2d_transpose_output = Conv2DTranspose(filters=self.colors, kernel_size=5,
                                                               strides=2, padding="same", activation=tf.tanh,
                                                               kernel_initializer=ort_init(),
                                                               bias_initializer=zeros_init())

    def _forward(self, **graph_nodes):
        msg = "G_SHAPE {}: {}"
        self.logger.debug(msg.format("in", self.z.shape))

        with tf.variable_scope("generator"):
            with tf.variable_scope("hidden1"):
                g_hidden1 = self.batch_normalization_1(self.dense_1(self.z), self.training)
                self.logger.debug(msg.format("gh1", g_hidden1.shape))
                g_hidden1 = tf.reshape(g_hidden1, [-1, self.h_dim * 4, 4, 4])
                self.logger.debug(msg.format("gh1", g_hidden1.shape))

            with tf.variable_scope("hidden2"):
                g_hidden1 = tf.transpose(g_hidden1, [0, 3, 2, 1])
                g_hidden2 = self.batch_normalization_2(self.conv2d_transpose_2(g_hidden1), self.training)
                self.logger.debug(msg.format("g_hidden2", g_hidden2.shape))

            with tf.variable_scope("hidden3"):
                g_hidden3 = self.batch_normalization_3(self.conv2d_transpose_3(g_hidden2))
                self.logger.debug(msg.format("g_hidden3", g_hidden3.shape))

            with tf.variable_scope("output"):
                g_out = self.conv2d_transpose_output(g_hidden3)
                self.logger.debug(msg.format("out", g_out.shape))

                # if experiment is binomial
                if self.colors == 1:
                    G_sample = self.binomial_sampling(g_out)
                else:
                    G_sample = self.multinomial_sampling(g_out)

        res = dict()
        res["generator"] = self
        res["z"] = self.z
        res["R"] = self.R
        res["G_output_logit"] = g_out
        res["G_sample"] = G_sample

        return res, dict(), dict()


class Gan60Generator(GanGenerator):

    def __init__(self, experiment):
        super().__init__(experiment)

        with tf.variable_scope("generator"):
            with tf.variable_scope("hidden1"):
                self.dense_1 = Dense(1024, activation=relu)
                self.batch_normalization_1 = BatchNormalization()
            with tf.variable_scope("hidden2"):
                self.dense_2 = Dense(self.h_dim * 2 * 5 * 5, activation=relu)
                self.batch_normalization_2 = BatchNormalization()
            with tf.variable_scope("hidden3"):
                self.conv2d_transpose_3 = Conv2DTranspose(filters=self.h_dim, kernel_size=5, strides=2,
                                                          padding="same", activation=relu,
                                                          kernel_initializer=ort_init(), bias_initializer=zeros_init())
                self.batch_normalization_3 = BatchNormalization()
            with tf.variable_scope("hidden4"):
                self.conv2d_transpose_4 = Conv2DTranspose(filters=self.h_dim, kernel_size=5, strides=2,
                                                          padding="same", activation=relu,
                                                          kernel_initializer=ort_init(), bias_initializer=zeros_init())
                self.batch_normalization_4 = BatchNormalization()
            with tf.variable_scope("output"):
                self.conv2d_transpose_output = Conv2DTranspose(filters=self.colors, kernel_size=5,
                                                               strides=3, padding="same", kernel_initializer=ort_init(),
                                                               bias_initializer=zeros_init())

    def _forward(self, **graph_nodes):
        msg = "G_SHAPE {}: {}"
        self.logger.debug(msg.format("in", self.z.shape))

        with tf.variable_scope("generator"):
            with tf.variable_scope("hidden1"):
                g_hidden1 = self.batch_normalization_1(self.dense_1(self.z))
                self.logger.debug(msg.format("gh1", g_hidden1.shape))

            with tf.variable_scope("hidden2"):
                g_hidden2 = self.batch_normalization_2(self.dense_2(g_hidden1))
                self.logger.debug(msg.format("gh2", g_hidden2.shape))
                g_hidden2 = tf.reshape(g_hidden2, [-1, self.h_dim * 2, 5, 5])
                self.logger.debug(msg.format("gh2", g_hidden2.shape))

            with tf.variable_scope("hidden3"):
                # deconv2d only supports nhwc channels. Transposing nchw to nhwc
                g_hidden2 = tf.transpose(g_hidden2, [0, 3, 2, 1])
                self.logger.debug(msg.format("gh3", g_hidden2.shape))
                g_hidden3 = self.batch_normalization_3(self.conv2d_transpose_3(g_hidden2))
                self.logger.debug(msg.format("gh3", g_hidden3.shape))

            with tf.variable_scope("hidden4"):
                g_hidden4 = self.batch_normalization_4(self.conv2d_transpose_4(g_hidden3))
                self.logger.debug(msg.format("gh4", g_hidden4.shape))

            with tf.variable_scope("output"):
                g_out = self.conv2d_transpose_output(g_hidden4)
                self.logger.debug(msg.format("out", g_out.shape))

                # if experiment is binomial
                if self.colors == 1:
                    G_sample = self.binomial_sampling(g_out)
                else:
                    G_sample = self.multinomial_sampling(g_out)

        res = dict()
        res["generator"] = self
        res["z"] = self.z
        res["R"] = self.R
        res["G_output_logit"] = g_out
        res["G_sample"] = G_sample

        return res, dict(), dict()
