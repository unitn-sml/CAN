"""
This files containts generic architectures that are not strictly related with the GAN / CAN training.
"""
import sys
import tensorflow as tf
sys.path.append("..")
from base_layers import Layer
import numpy as np
# using can repo standards for architectures


class _ReachabilityNetwork(Layer):

    def __init__(self, experiment, trainable=True):
        """
        Call the superclass constructor on experiment and initialize all the required layers.
        """
        super().__init__(experiment)
        if self.experiment:
            self.input_shape = [*self.experiment["SHAPE"][:2], 1]
        else:
            self.input_shape = (14, 28, 1)

        with tf.variable_scope('constraints'):
            with tf.variable_scope('reachability'):
                self.model = self._get_model(trainable=trainable)

    def _get_model(self, trainable):
        raise NotImplementedError("This method should be overridden by subclasses")

    def _pre_processing(self, **graph_nodes):
        self._requires('G_probab_solid', **graph_nodes)

    def _forward(self, **graph_nodes):
        """
        :param graph_nodes: Dict of tf graph nodes, G_probab_solid must be present.
        :return: A pair of dicts, where the first one is for giving nodes to the trainer,
        the second is for providing nodes that should be logged.
        """
        input_node = graph_nodes['G_probab_solid']

        res = dict()
        res["reachability_map"] = self.model(input_node)
        return res, dict(), dict()

    def _save_weights(self, path):
        self.model.save_weights(
            filepath=(path if path.endswith('.h5') else path + '.h5'))

    def _restore_weights(self, path):
        self.model.load_weights(path)

    def _init_weights(self, **graph_nodes):
        if self.model_path:
            self._restore_weights(self.model_path)


class ReachabilityNetwork4Layers(_ReachabilityNetwork):
    """
    Given a batch of levels in probability form (no logits), return the reachability map of each level.
    Each level should be of the form [height, width, 2], having on the last two channels the probabilities
    of that tile being air or solid.
    The output is the reachability distribution, with values close to 1 if the tile is reachable and values
    close to 0 otherwise.
    """

    def _get_model(self, trainable):
        return tf.keras.Sequential(
            layers=[
                tf.keras.layers.Conv2D(
                    filters=8,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding='SAME',
                    dilation_rate=(1, 1),
                    data_format='channels_last',
                    activation=tf.nn.relu,
                    input_shape=self.input_shape,
                    trainable=trainable
                ),
                tf.keras.layers.Conv2D(
                    filters=16,
                    kernel_size=(5, 5),
                    strides=(1, 1),
                    padding='SAME',
                    dilation_rate=(1, 1),
                    data_format='channels_last',
                    activation=tf.nn.relu,
                    trainable=trainable
                ),
                tf.keras.layers.Conv2D(
                    filters=32,
                    kernel_size=(7, 7),
                    strides=(1, 1),
                    padding='SAME',
                    dilation_rate=(1, 1),
                    data_format='channels_last',
                    activation=tf.nn.relu,
                    trainable=trainable
                ),
                tf.keras.layers.Conv2D(
                    filters=64,
                    kernel_size=(9, 9),
                    strides=(1, 1),
                    padding='SAME',
                    dilation_rate=(1, 1),
                    data_format='channels_last',
                    activation=tf.nn.relu,
                    trainable=trainable
                ),
                tf.keras.layers.Dense(
                    2,
                    activation=tf.nn.relu,
                    trainable=trainable
                )
            ]
        )


class ReachabilityNetwork5Layers(_ReachabilityNetwork):
    """
    Given a batch of levels in probability form (no logits), return the reachability map of each level.
    Each level should be of the form [height, width, 2], having on the last two channels the probabilities
    of that tile being air or solid.
    The output is the reachability distribution, with values close to 1 if the tile is reachable and values
    close to 0 otherwise.
    """

    def _get_model(self, trainable):
        return tf.keras.Sequential(
            layers=[
                tf.keras.layers.Conv2D(
                    filters=8,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding='SAME',
                    dilation_rate=(1, 1),
                    data_format='channels_last',
                    activation=tf.nn.relu,
                    input_shape=self.input_shape,
                    trainable=trainable
                ),
                tf.keras.layers.Conv2D(
                    filters=16,
                    kernel_size=(5, 5),
                    strides=(1, 1),
                    padding='SAME',
                    dilation_rate=(1, 1),
                    data_format='channels_last',
                    activation=tf.nn.relu,
                    trainable=trainable
                ),
                tf.keras.layers.Conv2D(
                    filters=32,
                    kernel_size=(7, 7),
                    strides=(1, 1),
                    padding='SAME',
                    dilation_rate=(1, 1),
                    data_format='channels_last',
                    activation=tf.nn.relu,
                    trainable=trainable
                ),
                tf.keras.layers.Conv2D(
                    filters=64,
                    kernel_size=(9, 9),
                    strides=(1, 1),
                    padding='SAME',
                    dilation_rate=(1, 1),
                    data_format='channels_last',
                    activation=tf.nn.relu,
                    trainable=trainable
                ),
                tf.keras.layers.Conv2D(
                    filters=128,
                    kernel_size=(11, 11),
                    strides=(1, 1),
                    padding='SAME',
                    dilation_rate=(1, 1),
                    data_format='channels_last',
                    activation=tf.nn.relu,
                    trainable=trainable
                ),
                tf.keras.layers.Dense(
                    2,
                    activation=tf.nn.relu,
                    trainable=trainable
                )            
            ]
        )

    
class ReachabilityNetwork6Layers(_ReachabilityNetwork):
    """
    Given a batch of levels in probability form (no logits), return the reachability map of each level.
    Each level should be of the form [height, width, 2], having on the last two channels the probabilities
    of that tile being air or solid.
    The output is the reachability distribution, with values close to 1 if the tile is reachable and values
    close to 0 otherwise.
    """

    def _get_model(self, trainable):
        return tf.keras.Sequential(
            layers=[
                tf.keras.layers.Conv2D(
                    filters=4,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding='SAME',
                    dilation_rate=(1, 1),
                    data_format='channels_last',
                    activation=tf.nn.relu,
                    input_shape=self.input_shape,
                    trainable=trainable
                ),
                tf.keras.layers.Conv2D(
                    filters=8,
                    kernel_size=(5, 5),
                    strides=(1, 1),
                    padding='SAME',
                    dilation_rate=(1, 1),
                    data_format='channels_last',
                    activation=tf.nn.relu,
                    trainable=trainable
                ),
                tf.keras.layers.Conv2D(
                    filters=16,
                    kernel_size=(7, 7),
                    strides=(1, 1),
                    padding='SAME',
                    dilation_rate=(1, 1),
                    data_format='channels_last',
                    activation=tf.nn.relu,
                    trainable=trainable
                ),
                tf.keras.layers.Conv2D(
                    filters=32,
                    kernel_size=(9, 9),
                    strides=(1, 1),
                    padding='SAME',
                    dilation_rate=(1, 1),
                    data_format='channels_last',
                    activation=tf.nn.relu,
                    trainable=trainable
                ),
                tf.keras.layers.Conv2D(
                    filters=64,
                    kernel_size=(11, 11),
                    strides=(1, 1),
                    padding='SAME',
                    dilation_rate=(1, 1),
                    data_format='channels_last',
                    activation=tf.nn.relu,
                    trainable=trainable
                ),
                tf.keras.layers.Conv2D(
                    filters=128,
                    kernel_size=(13, 13),
                    strides=(1, 1),
                    padding='SAME',
                    dilation_rate=(1, 1),
                    data_format='channels_last',
                    activation=tf.nn.relu,
                    trainable=trainable
                ),
                tf.keras.layers.Dense(
                    32,
                    activation=tf.nn.relu,
                    trainable=trainable
                ),
                tf.keras.layers.Dense(
                    2,
                    activation=tf.nn.relu,
                    trainable=trainable
                )
            ]
        )


class ReachabilityNetwork7Layers(_ReachabilityNetwork):
    """
    Given a batch of levels in probability form (no logits), return the reachability map of each level.
    Each level should be of the form [height, width, 2], having on the last two channels the probabilities
    of that tile being air or solid.
    The output is the reachability distribution, with values close to 1 if the tile is reachable and values
    close to 0 otherwise.
    """

    def _get_model(self, trainable):
        return tf.keras.Sequential(
            layers=[
                tf.keras.layers.Conv2D(
                    filters=4,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding='SAME',
                    dilation_rate=(1, 1),
                    data_format='channels_last',
                    activation=tf.nn.relu,
                    input_shape=self.input_shape,
                    trainable=trainable
                ),
                tf.keras.layers.Conv2D(
                    filters=8,
                    kernel_size=(5, 5),
                    strides=(1, 1),
                    padding='SAME',
                    dilation_rate=(1, 1),
                    data_format='channels_last',
                    activation=tf.nn.relu,
                    trainable=trainable
                ),
                tf.keras.layers.Conv2D(
                    filters=16,
                    kernel_size=(7, 7),
                    strides=(1, 1),
                    padding='SAME',
                    dilation_rate=(1, 1),
                    data_format='channels_last',
                    activation=tf.nn.relu,
                    trainable=trainable
                ),
                tf.keras.layers.Conv2D(
                    filters=32,
                    kernel_size=(9, 9),
                    strides=(1, 1),
                    padding='SAME',
                    dilation_rate=(1, 1),
                    data_format='channels_last',
                    activation=tf.nn.relu,
                    trainable=trainable
                ),
                tf.keras.layers.Conv2D(
                    filters=64,
                    kernel_size=(11, 11),
                    strides=(1, 1),
                    padding='SAME',
                    dilation_rate=(1, 1),
                    data_format='channels_last',
                    activation=tf.nn.relu,
                    trainable=trainable
                ),
                tf.keras.layers.Conv2D(
                    filters=128,
                    kernel_size=(13, 13),
                    strides=(1, 1),
                    padding='SAME',
                    dilation_rate=(1, 1),
                    data_format='channels_last',
                    activation=tf.nn.relu,
                    trainable=trainable
                ),
                tf.keras.layers.Conv2D(
                    filters=256,
                    kernel_size=(15, 15),
                    strides=(1, 1),
                    padding='SAME',
                    dilation_rate=(1, 1),
                    data_format='channels_last',
                    activation=tf.nn.relu,
                    trainable=trainable
                ),
                tf.keras.layers.Dense(
                    32,
                    activation=tf.nn.relu,
                    trainable=trainable
                ),
                tf.keras.layers.Dense(
                    2,
                    activation=tf.nn.relu,
                    trainable=trainable
                )
            ]
        )

    
class ReachabilityNetwork8Layers(_ReachabilityNetwork):
    """
    Given a batch of levels in probability form (no logits), return the reachability map of each level.
    Each level should be of the form [height, width, 2], having on the last two channels the probabilities
    of that tile being air or solid.
    The output is the reachability distribution, with values close to 1 if the tile is reachable and values
    close to 0 otherwise.
    """

    def _get_model(self, trainable):
        return tf.keras.Sequential(
            layers=[
                tf.keras.layers.Conv2D(
                    filters=2,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding='SAME',
                    dilation_rate=(1, 1),
                    data_format='channels_last',
                    activation=tf.nn.relu,
                    input_shape=self.input_shape,
                    trainable=trainable
                ),
                tf.keras.layers.Conv2D(
                    filters=4,
                    kernel_size=(5, 5),
                    strides=(1, 1),
                    padding='SAME',
                    dilation_rate=(1, 1),
                    data_format='channels_last',
                    activation=tf.nn.relu,
                    trainable=trainable
                ),
                tf.keras.layers.Conv2D(
                    filters=6,
                    kernel_size=(7, 7),
                    strides=(1, 1),
                    padding='SAME',
                    dilation_rate=(1, 1),
                    data_format='channels_last',
                    activation=tf.nn.relu,
                    trainable=trainable
                ),
                tf.keras.layers.Conv2D(
                    filters=10,
                    kernel_size=(9, 9),
                    strides=(1, 1),
                    padding='SAME',
                    dilation_rate=(1, 1),
                    data_format='channels_last',
                    activation=tf.nn.relu,
                    trainable=trainable
                ),
                tf.keras.layers.Conv2D(
                    filters=16,
                    kernel_size=(11, 11),
                    strides=(1, 1),
                    padding='SAME',
                    dilation_rate=(1, 1),
                    data_format='channels_last',
                    activation=tf.nn.relu,
                    trainable=trainable
                ),
                tf.keras.layers.Conv2D(
                    filters=20,
                    kernel_size=(13, 13),
                    strides=(1, 1),
                    padding='SAME',
                    dilation_rate=(1, 1),
                    data_format='channels_last',
                    activation=tf.nn.relu,
                    trainable=trainable
                ),
                tf.keras.layers.Conv2D(
                    filters=32,
                    kernel_size=(15, 15),
                    strides=(1, 1),
                    padding='SAME',
                    dilation_rate=(1, 1),
                    data_format='channels_last',
                    activation=tf.nn.relu,
                    trainable=trainable
                ),
                tf.keras.layers.Conv2D(
                    filters=64,
                    kernel_size=(17, 17),
                    strides=(1, 1),
                    padding='SAME',
                    dilation_rate=(1, 1),
                    data_format='channels_last',
                    activation=tf.nn.relu,
                    trainable=trainable
                ),
                tf.keras.layers.Dense(
                    16,
                    activation=tf.nn.relu,
                    trainable=trainable
                ),
                tf.keras.layers.Dense(
                    2,
                    activation=tf.nn.relu,
                    trainable=trainable
                )
            ]
        )


class ReachabilityNetwork8LayersDeep(_ReachabilityNetwork):
    """
    Given a batch of levels in probability form (no logits), return the reachability map of each level.
    Each level should be of the form [height, width, 2], having on the last two channels the probabilities
    of that tile being air or solid.
    The output is the reachability distribution, with values close to 1 if the tile is reachable and values
    close to 0 otherwise.
    """

    def _get_model(self, trainable):
        return tf.keras.Sequential(
            layers=[
                tf.keras.layers.Conv2D(
                    filters=8,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding='SAME',
                    dilation_rate=(1, 1),
                    data_format='channels_last',
                    activation=tf.nn.relu,
                    input_shape=self.input_shape,
                    trainable=trainable
                ),
                tf.keras.layers.Conv2D(
                    filters=16,
                    kernel_size=(5, 5),
                    strides=(1, 1),
                    padding='SAME',
                    dilation_rate=(1, 1),
                    data_format='channels_last',
                    activation=tf.nn.relu,
                    trainable=trainable
                ),
                tf.keras.layers.Conv2D(
                    filters=24,
                    kernel_size=(7, 7),
                    strides=(1, 1),
                    padding='SAME',
                    dilation_rate=(1, 1),
                    data_format='channels_last',
                    activation=tf.nn.relu,
                    trainable=trainable
                ),
                tf.keras.layers.Conv2D(
                    filters=32,
                    kernel_size=(9, 9),
                    strides=(1, 1),
                    padding='SAME',
                    dilation_rate=(1, 1),
                    data_format='channels_last',
                    activation=tf.nn.relu,
                    trainable=trainable
                ),
                tf.keras.layers.Conv2D(
                    filters=64,
                    kernel_size=(11, 11),
                    strides=(1, 1),
                    padding='SAME',
                    dilation_rate=(1, 1),
                    data_format='channels_last',
                    activation=tf.nn.relu,
                    trainable=trainable
                ),
                tf.keras.layers.Conv2D(
                    filters=96,
                    kernel_size=(13, 13),
                    strides=(1, 1),
                    padding='SAME',
                    dilation_rate=(1, 1),
                    data_format='channels_last',
                    activation=tf.nn.relu,
                    trainable=trainable
                ),
                tf.keras.layers.Conv2D(
                    filters=128,
                    kernel_size=(15, 15),
                    strides=(1, 1),
                    padding='SAME',
                    dilation_rate=(1, 1),
                    data_format='channels_last',
                    activation=tf.nn.relu,
                    trainable=trainable
                ),
                tf.keras.layers.Conv2D(
                    filters=192,
                    kernel_size=(17, 17),
                    strides=(1, 1),
                    padding='SAME',
                    dilation_rate=(1, 1),
                    data_format='channels_last',
                    activation=tf.nn.relu,
                    trainable=trainable
                ),
                tf.keras.layers.Dense(
                    32,
                    activation=tf.nn.relu,
                    trainable=trainable
                ),
                tf.keras.layers.Dense(
                    2,
                    activation=tf.nn.relu,
                    trainable=trainable
                )
            ]
        )


class ReachabilityNetwork8LayersFlat(_ReachabilityNetwork):
    """
    Given a batch of levels in probability form (no logits), return the reachability map of each level.
    Each level should be of the form [height, width, 2], having on the last two channels the probabilities
    of that tile being air or solid.
    The output is the reachability distribution, with values close to 1 if the tile is reachable and values
    close to 0 otherwise.
    """

    def _get_model(self, trainable):
        return tf.keras.Sequential(
            layers=[
                tf.keras.layers.Conv2D(
                    filters=32,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding='SAME',
                    dilation_rate=(1, 1),
                    data_format='channels_last',
                    activation=tf.nn.relu,
                    input_shape=self.input_shape,
                    trainable=trainable
                ),
                tf.keras.layers.Conv2D(
                    filters=32,
                    kernel_size=(5, 5),
                    strides=(1, 1),
                    padding='SAME',
                    dilation_rate=(1, 1),
                    data_format='channels_last',
                    activation=tf.nn.relu,
                    trainable=trainable
                ),
                tf.keras.layers.Conv2D(
                    filters=32,
                    kernel_size=(7, 7),
                    strides=(1, 1),
                    padding='SAME',
                    dilation_rate=(1, 1),
                    data_format='channels_last',
                    activation=tf.nn.relu,
                    trainable=trainable
                ),
                tf.keras.layers.Conv2D(
                    filters=32,
                    kernel_size=(9, 9),
                    strides=(1, 1),
                    padding='SAME',
                    dilation_rate=(1, 1),
                    data_format='channels_last',
                    activation=tf.nn.relu,
                    trainable=trainable
                ),
                tf.keras.layers.Conv2D(
                    filters=32,
                    kernel_size=(11, 11),
                    strides=(1, 1),
                    padding='SAME',
                    dilation_rate=(1, 1),
                    data_format='channels_last',
                    activation=tf.nn.relu,
                    trainable=trainable
                ),
                tf.keras.layers.Conv2D(
                    filters=32,
                    kernel_size=(13, 13),
                    strides=(1, 1),
                    padding='SAME',
                    dilation_rate=(1, 1),
                    data_format='channels_last',
                    activation=tf.nn.relu,
                    trainable=trainable
                ),
                tf.keras.layers.Conv2D(
                    filters=32,
                    kernel_size=(15, 15),
                    strides=(1, 1),
                    padding='SAME',
                    dilation_rate=(1, 1),
                    data_format='channels_last',
                    activation=tf.nn.relu,
                    trainable=trainable
                ),
                tf.keras.layers.Conv2D(
                    filters=32,
                    kernel_size=(17, 17),
                    strides=(1, 1),
                    padding='SAME',
                    dilation_rate=(1, 1),
                    data_format='channels_last',
                    activation=tf.nn.relu,
                    trainable=trainable
                ),
                tf.keras.layers.Dense(
                    2,
                    activation=tf.nn.relu,
                    trainable=trainable
                )
            ]
        )


class ReachabilityNetwork10Layers(_ReachabilityNetwork):
    """
    Given a batch of levels in probability form (no logits), return the reachability map of each level.
    Each level should be of the form [height, width, 2], having on the last two channels the probabilities
    of that tile being air or solid.
    The output is the reachability distribution, with values close to 1 if the tile is reachable and values
    close to 0 otherwise.
    """

    def _get_model(self, trainable):
        return tf.keras.Sequential(
            layers=[
                tf.keras.layers.Conv2D(
                    filters=2,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding='SAME',
                    dilation_rate=(1, 1),
                    data_format='channels_last',
                    activation=tf.nn.relu,
                    input_shape=self.input_shape,
                    trainable=trainable
                ),
                tf.keras.layers.Conv2D(
                    filters=4,
                    kernel_size=(5, 5),
                    strides=(1, 1),
                    padding='SAME',
                    dilation_rate=(1, 1),
                    data_format='channels_last',
                    activation=tf.nn.relu,
                    trainable=trainable
                ),
                tf.keras.layers.Conv2D(
                    filters=6,
                    kernel_size=(7, 7),
                    strides=(1, 1),
                    padding='SAME',
                    dilation_rate=(1, 1),
                    data_format='channels_last',
                    activation=tf.nn.relu,
                    trainable=trainable
                ),
                tf.keras.layers.Conv2D(
                    filters=8,
                    kernel_size=(9, 9),
                    strides=(1, 1),
                    padding='SAME',
                    dilation_rate=(1, 1),
                    data_format='channels_last',
                    activation=tf.nn.relu,
                    trainable=trainable
                ),
                tf.keras.layers.Conv2D(
                    filters=12,
                    kernel_size=(11, 11),
                    strides=(1, 1),
                    padding='SAME',
                    dilation_rate=(1, 1),
                    data_format='channels_last',
                    activation=tf.nn.relu,
                    trainable=trainable
                ),
                tf.keras.layers.Conv2D(
                    filters=16,
                    kernel_size=(13, 13),
                    strides=(1, 1),
                    padding='SAME',
                    dilation_rate=(1, 1),
                    data_format='channels_last',
                    activation=tf.nn.relu,
                    trainable=trainable
                ),
                tf.keras.layers.Conv2D(
                    filters=20,
                    kernel_size=(15, 15),
                    strides=(1, 1),
                    padding='SAME',
                    dilation_rate=(1, 1),
                    data_format='channels_last',
                    activation=tf.nn.relu,
                    trainable=trainable
                ),
                tf.keras.layers.Conv2D(
                    filters=32,
                    kernel_size=(17, 17),
                    strides=(1, 1),
                    padding='SAME',
                    dilation_rate=(1, 1),
                    data_format='channels_last',
                    activation=tf.nn.relu,
                    trainable=trainable
                ),
                tf.keras.layers.Conv2D(
                    filters=48,
                    kernel_size=(19, 19),
                    strides=(1, 1),
                    padding='SAME',
                    dilation_rate=(1, 1),
                    data_format='channels_last',
                    activation=tf.nn.relu,
                    trainable=trainable
                ),
                tf.keras.layers.Conv2D(
                    filters=64,
                    kernel_size=(21, 21),
                    strides=(1, 1),
                    padding='SAME',
                    dilation_rate=(1, 1),
                    data_format='channels_last',
                    activation=tf.nn.relu,
                    trainable=trainable
                ),
                tf.keras.layers.Dense(
                    32,
                    activation=tf.nn.relu,
                    trainable=trainable
                ),
                tf.keras.layers.Dense(
                    8,
                    activation=tf.nn.relu,
                    trainable=trainable
                ),
                tf.keras.layers.Dense(
                    2,
                    activation=tf.nn.relu,
                    trainable=trainable
                )
            ]
        )


class ReachabilityNetwork12Layers(_ReachabilityNetwork):
    """
    Given a batch of levels in probability form (no logits), return the reachability map of each level.
    Each level should be of the form [height, width, 2], having on the last two channels the probabilities
    of that tile being air or solid.
    The output is the reachability distribution, with values close to 1 if the tile is reachable and values
    close to 0 otherwise.
    """

    def _get_model(self, trainable):
        return tf.keras.Sequential(
            layers=[
                tf.keras.layers.Conv2D(
                    filters=2,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding='SAME',
                    dilation_rate=(1, 1),
                    data_format='channels_last',
                    activation=tf.nn.relu,
                    input_shape=self.input_shape,
                    trainable=trainable
                ),
                tf.keras.layers.Conv2D(
                    filters=4,
                    kernel_size=(5, 5),
                    strides=(1, 1),
                    padding='SAME',
                    dilation_rate=(1, 1),
                    data_format='channels_last',
                    activation=tf.nn.relu,
                    trainable=trainable
                ),
                tf.keras.layers.Conv2D(
                    filters=6,
                    kernel_size=(7, 7),
                    strides=(1, 1),
                    padding='SAME',
                    dilation_rate=(1, 1),
                    data_format='channels_last',
                    activation=tf.nn.relu,
                    trainable=trainable
                ),
                tf.keras.layers.Conv2D(
                    filters=8,
                    kernel_size=(9, 9),
                    strides=(1, 1),
                    padding='SAME',
                    dilation_rate=(1, 1),
                    data_format='channels_last',
                    activation=tf.nn.relu,
                    trainable=trainable
                ),
                tf.keras.layers.Conv2D(
                    filters=12,
                    kernel_size=(11, 11),
                    strides=(1, 1),
                    padding='SAME',
                    dilation_rate=(1, 1),
                    data_format='channels_last',
                    activation=tf.nn.relu,
                    trainable=trainable
                ),
                tf.keras.layers.Conv2D(
                    filters=16,
                    kernel_size=(13, 13),
                    strides=(1, 1),
                    padding='SAME',
                    dilation_rate=(1, 1),
                    data_format='channels_last',
                    activation=tf.nn.relu,
                    trainable=trainable
                ),
                tf.keras.layers.Conv2D(
                    filters=20,
                    kernel_size=(15, 15),
                    strides=(1, 1),
                    padding='SAME',
                    dilation_rate=(1, 1),
                    data_format='channels_last',
                    activation=tf.nn.relu,
                    trainable=trainable
                ),
                tf.keras.layers.Conv2D(
                    filters=32,
                    kernel_size=(17, 17),
                    strides=(1, 1),
                    padding='SAME',
                    dilation_rate=(1, 1),
                    data_format='channels_last',
                    activation=tf.nn.relu,
                    trainable=trainable
                ),
                tf.keras.layers.Conv2D(
                    filters=48,
                    kernel_size=(19, 19),
                    strides=(1, 1),
                    padding='SAME',
                    dilation_rate=(1, 1),
                    data_format='channels_last',
                    activation=tf.nn.relu,
                    trainable=trainable
                ),
                tf.keras.layers.Conv2D(
                    filters=64,
                    kernel_size=(21, 21),
                    strides=(1, 1),
                    padding='SAME',
                    dilation_rate=(1, 1),
                    data_format='channels_last',
                    activation=tf.nn.relu,
                    trainable=trainable
                ),
                tf.keras.layers.Conv2D(
                    filters=96,
                    kernel_size=(23, 23),
                    strides=(1, 1),
                    padding='SAME',
                    dilation_rate=(1, 1),
                    data_format='channels_last',
                    activation=tf.nn.relu,
                    trainable=trainable
                ),
                tf.keras.layers.Conv2D(
                    filters=128,
                    kernel_size=(25, 25),
                    strides=(1, 1),
                    padding='SAME',
                    dilation_rate=(1, 1),
                    data_format='channels_last',
                    activation=tf.nn.relu,
                    trainable=trainable
                ),
                tf.keras.layers.Dense(
                    64,
                    activation=tf.nn.relu,
                    trainable=trainable
                ),
                tf.keras.layers.Dense(
                    16,
                    activation=tf.nn.relu,
                    trainable=trainable
                ),
                tf.keras.layers.Dense(
                    8,
                    activation=tf.nn.relu,
                    trainable=trainable
                ),
                tf.keras.layers.Dense(
                    2,
                    activation=tf.nn.relu,
                    trainable=trainable
                )
            ]
        )


class ReachabilityProduct(tf.keras.layers.Layer):

    def __init__(self, double=False):
        super(ReachabilityProduct, self).__init__()
        self.paddings = np.array([
            [0, 0],
            [0, 1],
            [0, 0],
            [0, 0]
        ])
        self.double = double

    def call(self, inputs):
        assert len(inputs.shape.as_list()) == 4, "ReachabilityProduct expects an input with shape [batch_size, height, width, 1]"
        
        solid = tf.pad(inputs[:, 1:, :, :], self.paddings, mode='constant', constant_values=0)
        air = 1 - inputs
        # create matrix of probabilities to be a stationary point
        stationary = tf.math.multiply(air, solid)
        if self.double:
            stationary = tf.concat([stationary, solid], axis=-1)
        return stationary


class ReachabilityNetworkSpecialLayers(_ReachabilityNetwork):
    """
    Given a batch of levels in probability form (no logits), return the reachability map of each level.
    Each level should be of the form [height, width, 2], having on the last two channels the probabilities
    of that tile being air or solid.
    The output is the reachability distribution, with values close to 1 if the tile is reachable and values
    close to 0 otherwise.
    """

    def _get_model(self, trainable):
        return tf.keras.Sequential(
            layers=[
                ReachabilityProduct(
                    double=True
                ),
                tf.keras.layers.Conv2D(
                    filters=8,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding='SAME',
                    dilation_rate=(1, 1),
                    data_format='channels_last',
                    activation=tf.nn.relu,
                    input_shape=self.input_shape,
                    trainable=trainable
                ),
                tf.keras.layers.Conv2D(
                    filters=16,
                    kernel_size=(5, 5),
                    strides=(1, 1),
                    padding='SAME',
                    dilation_rate=(1, 1),
                    data_format='channels_last',
                    activation=tf.nn.relu,
                    trainable=trainable
                ),
                tf.keras.layers.Conv2D(
                    filters=24,
                    kernel_size=(7, 7),
                    strides=(1, 1),
                    padding='SAME',
                    dilation_rate=(1, 1),
                    data_format='channels_last',
                    activation=tf.nn.relu,
                    trainable=trainable
                ),
                tf.keras.layers.Conv2D(
                    filters=32,
                    kernel_size=(9, 9),
                    strides=(1, 1),
                    padding='SAME',
                    dilation_rate=(1, 1),
                    data_format='channels_last',
                    activation=tf.nn.relu,
                    trainable=trainable
                ),
                tf.keras.layers.Conv2D(
                    filters=64,
                    kernel_size=(11, 11),
                    strides=(1, 1),
                    padding='SAME',
                    dilation_rate=(1, 1),
                    data_format='channels_last',
                    activation=tf.nn.relu,
                    trainable=trainable
                ),
                tf.keras.layers.Conv2D(
                    filters=96,
                    kernel_size=(13, 13),
                    strides=(1, 1),
                    padding='SAME',
                    dilation_rate=(1, 1),
                    data_format='channels_last',
                    activation=tf.nn.relu,
                    trainable=trainable
                ),
                tf.keras.layers.Conv2D(
                    filters=128,
                    kernel_size=(15, 15),
                    strides=(1, 1),
                    padding='SAME',
                    dilation_rate=(1, 1),
                    data_format='channels_last',
                    activation=tf.nn.relu,
                    trainable=trainable
                ),
                tf.keras.layers.Conv2D(
                    filters=192,
                    kernel_size=(17, 17),
                    strides=(1, 1),
                    padding='SAME',
                    dilation_rate=(1, 1),
                    data_format='channels_last',
                    activation=tf.nn.relu,
                    trainable=trainable
                ),
                tf.keras.layers.Dense(
                    32,
                    activation=tf.nn.relu,
                    trainable=trainable
                ),
                tf.keras.layers.Dense(
                    2,
                    activation=tf.nn.relu,
                    trainable=trainable
                )
            ]
        )
