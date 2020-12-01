import tensorflow as tf
from base_layers import Solver

"""
Generic Level Solver
"""
class _LevelDiscriminatorSolver(Solver):
    def __init__(self, experiment):
        super().__init__(experiment)
        self.minimizer = None

    def _pre_processing(self, **graph_nodes):
        assert "D_loss" in graph_nodes

    def _forward(self, params, **graph_nodes):
        D_loss = graph_nodes["D_loss"]
        D_solver = self.minimizer.minimize(D_loss, var_list=params, global_step=graph_nodes.get("D_global_step", None))
        return D_solver


class _LevelGeneratorSolver(Solver):
    def __init__(self, experiment):
        super().__init__(experiment)
        self.minimizer = None

    def _pre_processing(self, **graph_nodes):
        assert "G_loss" in graph_nodes

    def _forward(self, params, **graph_nodes):
        G_loss = graph_nodes["G_loss"]
        G_solver = self.minimizer.minimize(G_loss, var_list=params, global_step=graph_nodes.get("G_global_step", None))
        return G_solver


"""
Adam solvers for the generator and the discriminator. Should only define a self.minimizer private field.
"""

class LevelGeneratorAdamSolver(_LevelGeneratorSolver):

    def __init__(self, experiment):
        super().__init__(experiment)
        self.minimizer = tf.train.AdamOptimizer(learning_rate=experiment["LEARNING_RATE"], beta1=0.5, beta2=0.999)


class LevelDiscriminatorAdamSolver(_LevelDiscriminatorSolver):

    def __init__(self, experiment):
        super().__init__(experiment)
        self.minimizer = tf.train.AdamOptimizer(learning_rate=experiment["LEARNING_RATE"], beta1=0.5, beta2=0.999)


"""
RMSProps solvers for the generator and the discriminator
"""
    
class LevelGeneratorRMSPropSolver(_LevelGeneratorSolver):
    # TODO: maybe try tf.keras.optimizers.RMSprop? Why do pytorch, tf, keras and coffee use different standard implementations????
    def __init__(self, experiment):
        super().__init__(experiment)
        self.minimizer = tf.train.RMSPropOptimizer(learning_rate=experiment["LEARNING_RATE"]) #, decay=0.99, momentum=0.0, epsilon=1e-08, centered=False) seems to worsen performances


class LevelDiscriminatorRMSPropSolver(_LevelDiscriminatorSolver):

    def __init__(self, experiment):
        super().__init__(experiment)
        self.minimizer = tf.train.RMSPropOptimizer(learning_rate=experiment["LEARNING_RATE"]) #, decay=0.99, momentum=0.0, epsilon=1e-08, centered=False) seems to worsen performances
