import tensorflow as tf
from polygons_floor.generators import GanGenerator
from base_layers import Generator

from tensorflow import (orthogonal_initializer as ort_init, zeros_initializer as zeros_init)
from tensorflow.python.ops.nn import relu

from tensorflow.keras.layers import Layer, Dense, BatchNormalization, Conv2DTranspose


class GanGumbleGenerator(Generator):
    """
    Simply a generator with a R placeholder for the uniform random noise
    to be used by the gumble trick or concrete distribution.
    """

    def __init__(self, experiment):
        super().__init__(experiment)
        self.logger = experiment["LOGGER"]
        self.h_dim = experiment["H_DIM"]
        self.output_channels = experiment["SHAPE"][-1]

        if self.experiment["NUM_BGAN_SAMPLES"] is None:
            self.R = tf.placeholder(tf.float32, [self.experiment["BATCH_SIZE"]] + self.experiment["SHAPE"],
                                    "R_placeholder")

        else:
            self.R = tf.placeholder(
                tf.float32, [self.experiment["NUM_BGAN_SAMPLES"], self.experiment["BATCH_SIZE"]] +
                            self.experiment["SHAPE"], "R_placeholder")
        self.training = True


class GanGumbleBinomialGenerator(GanGumbleGenerator):
    def __init__(self, experiment):
        super().__init__(experiment)

    """
    See concrete distribution in
    THE CONCRETE DISTRIBUTION : A CONTINUOUS RELAXATION OF DISCRETE RANDOM VARIABLES

    """

    def _uniform_to_logistic(self, random_unif, eps=1e-20):
        return tf.log(random_unif + eps) - tf.log(1. - random_unif + eps)

    def _concrete_sample(self, random_unif, logits, temperature):
        """ Draw a sample from the Concrete distribution"""
        y = self._uniform_to_logistic(random_unif) + logits
        y = y / temperature
        return tf.sigmoid(y)

    def concrete(self, random_unif, logits, temperature, hard=False):
        y = self._concrete_sample(random_unif, logits, temperature)

        if hard:
            # not really working, need to fix/improve
            y_hard = tf.cast(y > 0.5, y.dtype)
            y = tf.stop_gradient(y_hard - y) + y
        return y


class GanGumbleMultinomialGenerator(GanGumbleGenerator):
    def __init__(self, experiment):
        super().__init__(experiment)

    """
    Stuff for the gumbel softmax (+ ST), code by
    https://gist.github.com/ericjang/1001afd374c2c3b7752545ce6d9ed349#file-gumbel-softmax-py
    (author of one of the two papers)
    There are some minor changes,
    related to allowing to input the noise, which we need in this repository due to how some constraints
    are computed in between steps for some architectures (can with features fed to D).
    """

    def _uniform_to_gumbel(self, random_unif, eps=1e-20):
        return -tf.log(-tf.log(random_unif + eps) + eps)

    def _gumbel_softmax_sample(self, random_unif, logits, temperature):
        """ Draw a sample from the Gumbel-Softmax distribution"""
        y = logits + self._uniform_to_gumbel(random_unif)
        return tf.nn.softmax(y / temperature)

    def gumbel_softmax(self, random_unif, logits, temperature, hard=False):
        """Sample from the Gumbel-Softmax distribution over the last dimension and optionally discretize.
        Args:
          random_unif: random uniform values between [0-1] of the same shape of the logits
          logits: unnormalized log-probs
          temperature: non-negative scalar
          hard: if True, take argmax, but differentiate w.r.t. soft sample y
        Returns:
          Sample from the Gumbel-Softmax distribution of the shape of the logits/random unif noise.
          If hard=True, then the returned sample will be one-hot, otherwise it will
          be a probabilitiy distribution that sums to 1 across classes
        """
        y = self._gumbel_softmax_sample(random_unif, logits, temperature)
        if hard:
            y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, -1, keep_dims=True)), y.dtype)
            y = tf.stop_gradient(y_hard - y) + y
        return y


class Gan20GumbleBinomialGenerator(GanGumbleBinomialGenerator):

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
                                                          padding="same",
                                                          activation=relu, kernel_initializer=ort_init(),
                                                          bias_initializer=zeros_init())
                self.batch_normalization_3 = BatchNormalization()
            with tf.variable_scope("output"):
                self.conv2d_transpose_output = Conv2DTranspose(filters=self.output_channels, kernel_size=5, strides=2,
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

                # check the gumbel papers for good temperature values
                G_sample = self.concrete(self.R, g_out, temperature=2. / 3., hard=self.experiment["HARD_SAMPLING"])

        res = dict()
        res["generator"] = self
        res["z"] = self.z
        res["R"] = self.R
        res["G_output_logit"] = G_sample
        res["G_sample"] = G_sample

        return res, dict(), dict()



class GanFormulaGumbleBinomialGenerator(GanGumbleBinomialGenerator):

    def __init__(self, experiment):
        super().__init__(experiment)
        self.sample_n_vars = \
            self.experiment["SHAPE"][0]

        with tf.variable_scope("generator"):
            with tf.variable_scope("hidden1"):
                self.dense_1 = Dense(self.h_dim, activation=relu)
                self.batch_normalization_1 = BatchNormalization()
            with tf.variable_scope("hidden2"):
                self.dense_2 = Dense(self.h_dim * 2, activation=relu)
                self.batch_normalization_2 = BatchNormalization()
            with tf.variable_scope("hidden3"):
                self.dense_3 = Dense(self.h_dim * 4, activation=relu)
                self.batch_normalization_3 = BatchNormalization()
            with tf.variable_scope("hidden4"):
                self.dense_4 = Dense(self.h_dim * 2, activation=relu)
                self.batch_normalization_4 = BatchNormalization()
            with tf.variable_scope("output"):
                self.dense_output = Dense(self.sample_n_vars, activation=relu)

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
            with tf.variable_scope("hidden4"):
                g_hidden4 = self.batch_normalization_4(self.dense_4(g_hidden3),
                                                       self.training)
                self.logger.debug(msg.format("gh4", g_hidden4.shape))
            with tf.variable_scope("output"):
                g_out = self.dense_output(g_hidden4)
                self.logger.debug(msg.format("out", g_out.shape))

                # check the gumbel papers for good temperature values
                G_sample = self.concrete(self.R, g_out, temperature=2. / 3., hard=self.experiment["HARD_SAMPLING"])

        res = dict()
        res["generator"] = self
        res["z"] = self.z
        res["R"] = self.R
        res["G_output_logit"] = G_sample
        res["G_sample"] = G_sample

        return res, dict(), dict()




class Gan20GumbleMultinomialGenerator(GanGumbleMultinomialGenerator):

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
                self.conv2d_transpose_output = Conv2DTranspose(filters=self.output_channels, kernel_size=5, strides=2,
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

                # check the gumbel papers for good temperature values
                G_sample = self.gumbel_softmax(self.R, g_out, temperature=2. / 3.,
                                               hard=self.experiment["HARD_SAMPLING"])

        res = dict()
        res["generator"] = self
        res["z"] = self.z
        res["R"] = self.R
        res["G_output_logit"] = g_out
        res["G_sample"] = G_sample

        return res, dict(), dict()


class Gan28GumbleBinomialGenerator(GanGumbleBinomialGenerator):

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
                self.conv2d_transpose_3 = Conv2DTranspose(filters=self.h_dim * 4, kernel_size=4, strides=2,
                                                          padding="same", activation=relu,
                                                          kernel_initializer=ort_init(), bias_initializer=zeros_init())
                self.batch_normalization_4 = BatchNormalization()

            with tf.variable_scope("output"):
                self.conv2d_transpose_output = Conv2DTranspose(filters=self.output_channels, kernel_size=4, strides=2,
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

                # check the gumbel papers for good temperature values
                G_sample = self.concrete(self.R, g_out, temperature=2. / 3., hard=self.experiment["HARD_SAMPLING"])

        res = dict()
        res["generator"] = self
        res["z"] = self.z
        res["R"] = self.R
        res["G_output_logit"] = G_sample
        res["G_sample"] = G_sample

        return res, dict(), dict()


class Gan20DGenerator(GanGenerator):

    def __init__(self, experiment):
        super().__init__(experiment)
        self.Dz = tf.placeholder(tf.float32, [None, 377], "Dz_placeholder")

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
                g_hidden1 = self.batch_normalization_1(self.dense_1(tf.concat([self.z, self.Dz], axis=1)),
                                                       self.training)
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
        res["z"] = self.z
        res["Dz"] = self.Dz
        res["R"] = self.R
        res["G_output_logit"] = g_out
        res["G_sample"] = G_sample

        return res, dict(), dict()


class Gan28GumbleBinomialGenerator2(GanGumbleBinomialGenerator):

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
                self.conv2d_transpose_3 = Conv2DTranspose(filters=self.h_dim * 4, kernel_size=4, strides=2,
                                                          padding="same", activation=relu,
                                                          kernel_initializer=ort_init(), bias_initializer=zeros_init())
                self.batch_normalization_4 = BatchNormalization()

            with tf.variable_scope("hidden5"):
                self.conv2d_transpose_4 = Conv2DTranspose(filters=self.h_dim * 2, kernel_size=4, strides=2,
                                                          padding="same", kernel_initializer=ort_init(),
                                                          activation=relu,
                                                          bias_initializer=zeros_init())
                self.batch_normalization_5 = BatchNormalization()

            with tf.variable_scope("output"):
                self.conv2d_transpose_output = Conv2DTranspose(filters=self.output_channels, kernel_size=1, strides=1,
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

            with tf.variable_scope("hidden5"):
                g_hidden5 = self.batch_normalization_5(self.conv2d_transpose_4(g_hidden4), self.training)
                self.logger.debug(msg.format("gh5", g_hidden5.shape))

            with tf.variable_scope("output"):
                g_out = self.conv2d_transpose_output(g_hidden5)
                self.logger.debug(msg.format("out", g_out.shape))

                if self.experiment["SHAPE"] == [26,26,1]:
                    # fast way to cut the borders around the generated sample to
                    # allow experimenting with shape 26x26 without changing the architecture
                    g_out = g_out[:, 1:27, 1:27]

                # check the gumbel papers for good temperature values
                G_sample = self.concrete(self.R, g_out, temperature=2. / 3., hard=self.experiment["HARD_SAMPLING"])

        res = dict()
        res["generator"] = self
        res["z"] = self.z
        res["R"] = self.R
        res["G_output_logit"] = G_sample
        res["G_sample"] = G_sample

        return res, dict(), dict()


class Gan28GumbleBinomialGenerator3(GanGumbleBinomialGenerator):

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
                self.conv2d_transpose_3 = Conv2DTranspose(filters=self.h_dim * 4, kernel_size=4, strides=4,
                                                          padding="same", activation=relu,
                                                          kernel_initializer=ort_init(), bias_initializer=zeros_init())
                self.batch_normalization_4 = BatchNormalization()

            with tf.variable_scope("output"):
                self.conv2d_transpose_output = Conv2DTranspose(filters=self.output_channels, kernel_size=1, strides=1,
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

                # check the gumbel papers for good temperature values
                G_sample = self.concrete(self.R, g_out, temperature=2. / 3., hard=self.experiment["HARD_SAMPLING"])

        res = dict()
        res["generator"] = self
        res["z"] = self.z
        res["R"] = self.R
        res["G_output_logit"] = G_sample
        res["G_sample"] = G_sample

        return res, dict(), dict()
