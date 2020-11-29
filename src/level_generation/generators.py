"""
File containing the level generator architectures.
"""
import tensorflow as tf
#import tensorflow_probability as tfp
from base_layers import Generator

from tensorflow.initializers import random_normal
from tensorflow import (orthogonal_initializer as ort_init, zeros_initializer as zeros_init, expand_dims)
from tensorflow.keras.layers import Layer, Dense, BatchNormalization, Conv2DTranspose, ReLU
from level_generation.support import _soft_onehotcategorical

class DCGAN_G(Generator):

    def __init__(self, experiment):
        super().__init__(experiment)
        
        self.logger = self.experiment["LOGGER"]
        self.shape = self.experiment['SHAPE']
        self.batch_size = self.experiment["BATCH_SIZE"]
        self.bgan_samples = self.experiment["NUM_BGAN_SAMPLES"]
        self.isize = self.experiment['ISIZE']
        self.number_filters = self.experiment['NUMBER_FILTERS_GENERATOR']
        self.normal_initializer = random_normal(mean=0.0, stddev=0.02)
        self.affine_normal_initializer = random_normal(mean=1.0, stddev=0.02)
        self.zero_initializer = tf.zeros_initializer()
        self.beta_sample = self.experiment["BETA_SAMPLE"]

        self.batch_norm_args = {
            'axis': -1,
            'momentum': 0.9,
            'epsilon': 1e-5,
            'gamma_initializer': self.affine_normal_initializer,
            'beta_initializer': self.zero_initializer,
        }

        assert self.isize % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize = self.number_filters // 2, 4
        while tisize != self.isize:
            cngf = cngf * 2
            tisize = tisize * 2

            self.deconv_1 = Conv2DTranspose(filters=cngf, 
                                            kernel_size=(4, 4),
                                            strides=(1, 1),
                                            padding="valid", 
                                            use_bias=False,
                                            kernel_initializer=self.normal_initializer)
            self.batch_normalization_1 = BatchNormalization(**self.batch_norm_args)
            self.relu_1 = ReLU()

            csize, cndf = 4, cngf
            self.layers_list = []
            while csize < self.isize//2:
                self.layers_list.append(Conv2DTranspose(filters=cngf//2,
                                                        kernel_size=(4,4),
                                                        strides=(2,2),
                                                        padding='SAME',
                                                        use_bias=False,
                                                        kernel_initializer=self.normal_initializer))
                self.layers_list.append(BatchNormalization(**self.batch_norm_args))
                self.layers_list.append(ReLU())
                cngf = cngf // 2
                csize = csize * 2

            self.deconv_final = Conv2DTranspose(self.shape[2],
                                                kernel_size=(4,4),
                                                strides=(2,2),
                                                padding='SAME',
                                                use_bias=False,
                                                kernel_initializer=self.normal_initializer)
            self.relu_final = ReLU()

    def _forward(self, **graph_nodes):
        msg = "G_SHAPE {}: {}"
        self.logger.debug(msg.format("in", self.z.shape))

        # input with shape (batch_size, z_dim)
        with tf.variable_scope("generator"):
            with tf.variable_scope("deconv_1"):
                x = expand_dims(self.z, axis=1)
                x = expand_dims(x, axis=1)
                # x.shape: (batch_size, 1, 1, z_dim=32)
                self.logger.debug(msg.format("in_expanded", x.shape))
                x = self.deconv_1(x)
                self.logger.debug(msg.format("deconv_1", x.shape))
                x = self.batch_normalization_1(x, training=True)
                self.logger.debug(msg.format("batchnorm_1", x.shape))
                x = self.relu_1(x)
                self.logger.debug(msg.format("relu_1", x.shape))
                # x.shape: (batch_size, 4, 4, 256)

            with tf.variable_scope("deconv_pyramind"):
                for layer in self.layers_list:
                    if layer.__class__.__name__ == 'BatchNormalizationV1':
                        x = layer(x, training=True)
                    else:
                        x = layer(x)
                    self.logger.debug(msg.format(layer.__class__.__name__, x.shape))
                # x.shape: (batch_size, 16, 16, 64)

            with tf.variable_scope("deconv_final"):
                x = self.deconv_final(x)
                self.logger.debug(msg.format("deconv_final", x.shape))
                g_output_logit = self.relu_final(x)
                self.logger.debug(msg.format("g_out", g_output_logit.shape))
                # shape: (batch_size, 32, 32, 13)
            
            with tf.variable_scope("output"):
                # cropping to level size
                g_output_logit = g_output_logit[:, :14, :28, :]
                probabs = tf.nn.softmax(g_output_logit, axis=-1)
                probabs_for_sampling = _soft_onehotcategorical(g_output_logit, beta=self.beta_sample)
                g_sample = tf.contrib.distributions.OneHotCategorical(probs=probabs_for_sampling).sample()

        res = dict()
        res["z"] = self.z
        res["G_probs"] = probabs
        res["G_sample"] = g_sample
        res['G_output_logit'] = g_output_logit

        return res, dict(), dict()


