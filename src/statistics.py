"""
Common statistics module

"""

from base_layers import Statistic
import tensorflow as tf


class GeneratorGradient(Statistic):

    def _pre_processing(self, **graph_nodes):
        assert "G_loss" in graph_nodes

    def _forward(self, **graph_nodes):
        G_loss = graph_nodes["G_loss"]

        with tf.variable_scope("generator", reuse=True):
            theta_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "can/generator")
            grads = tf.gradients(G_loss, theta_G)
            grads = [tf.reduce_mean(tf.abs(g)) for g in grads]
            grads = tf.stack(grads, axis=0)
            grads = tf.reduce_mean(grads)
            summary = tf.summary.scalar("mean_gradient", grads)

        return tf.summary.merge([summary])


class DiscriminatorGradient(Statistic):

    def _pre_processing(self, **graph_nodes):
        assert "D_loss" in graph_nodes

    def _forward(self, **graph_nodes):
        D_loss = graph_nodes["D_loss"]

        with tf.variable_scope("discriminator", reuse=True):
            theta_D = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "can/discriminator")
            grads = tf.gradients(D_loss, theta_D)
            grads = [tf.reduce_mean(tf.abs(g)) for g in grads]
            grads = tf.stack(grads, axis=0)
            grads = tf.reduce_mean(grads)
            summary = tf.summary.scalar("mean_gradient", grads)

        return tf.summary.merge([summary])
