"""
This is the common solvers file. Here we have functions
that given experiment, losses and parameters return the optimizer
for both the discriminator and the generator.
"""
import tensorflow as tf
from base_layers import Solver


class DiscriminatorAdamSolver(Solver):
    def __init__(self, experiment):
        super().__init__(experiment)
        self.minimizer = tf.train.AdamOptimizer(self.experiment["DISCRIMINATOR_LEARNING_RATE"], beta1=0.5)

    def _pre_processing(self, **graph_nodes):
        assert "D_loss" in graph_nodes

    def _forward(self, params, **graph_nodes):
        D_loss = graph_nodes["D_loss"]
        D_solver = self.minimizer.minimize(D_loss, var_list=params, global_step=graph_nodes.get("D_global_step", None))
        return D_solver


class GeneratorAdamSolver(Solver):
    def __init__(self, experiment):
        super().__init__(experiment)
        self.minimizer = tf.train.AdamOptimizer(self.experiment["GENERATOR_LEARNING_RATE"], beta1=0.5)

    def _pre_processing(self, **graph_nodes):
        assert "G_loss" in graph_nodes

    def _forward(self, params, **graph_nodes):
        G_loss = graph_nodes["G_loss"]

        self.experiment["LOGGER"].debug(
            "Adding batch normalization moving_mean and moving_variance to the "
            "optimizer for generator")
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):
            G_solver = self.minimizer.minimize(G_loss, var_list=params, global_step=graph_nodes.get("G_global_step", None))

        return G_solver
