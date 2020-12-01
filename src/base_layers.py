"""
Base Layer class. All the generators, discriminators, losses and solver should first subclass the Layer class.
"""
import tensorflow as tf


class Layer:

    def __init__(self, experiment, *args, **kwargs):
        """
        init here all the layers you will need
        :param experiment: the experiment instance containing all the configuration parameters
        """
        self.experiment = experiment

    def __call__(self, **graph_nodes):
        """
        this method is called when the layer is called as a function.
        the first call to _pre_processing will make the necessary assertions to be sure that
        this class is called correctly.
        then the _forward method will do the computations on the input tensors and return
        the results.
        if needed, some postprocessing operations can be inserted in the _post_processing method

        :param **graph_nodes: the dict of important nodes that has to be transformed
        :return: the results
        """
        self._pre_processing(**graph_nodes)
        new_nodes, nodes_to_log, nodes_to_init = self._forward(**graph_nodes)
        self._post_processing(new_nodes, nodes_to_log, nodes_to_init)

        return (new_nodes, nodes_to_log, nodes_to_init)

    def _init_weights(self, **graph_nodes):
        """
        do some initialization of weights, load some models from file.
        this method is called after having done the global initialization step, so your
        models and weights will not be overridden by default initialization session.
        """
        pass

    def _pre_processing(self, **graph_nodes):
        """
        basically list all the nodes that are required by this layer, checking
        that their key is in graph_nodes
        :param graph_nodes: the dict of important nodes
        """
        pass

    def _requires(self, key, **graph_nodes):
        """
        basic pattern to check that a specific node is contained in graph_nodes.
        """
        if not isinstance(key, list):
            key = [key]
        for k in key:
            assert k in graph_nodes, "{} is required by {}".format(k, self.__class__.__name__)

    def _post_processing(self, new_nodes, nodes_to_log, nodes_to_init):
        """
        used to do some post processing.
        :param graph_nodes.
        """
        pass

    def _forward(self, **graph_nodes):
        """
        it applies a list of ops on some of the important nodes in graph_nodes
        and returns some results
        :param **graph_nodes: the dict of important nodes
        :return:
        """
        raise NotImplementedError("This method should be overridden by children")


########################################################################################################################
############################################## GENERATORS ##############################################################
########################################################################################################################

class Generator(Layer):

    def __init__(self, experiment, *args, **kwargs):
        super().__init__(experiment, *args, **kwargs)
        """
        init here all the layers you will need
        :param experiment: the experiment instance containing all the configuration parameters
        """
        self.z = tf.placeholder(tf.float32, [None, experiment["Z_DIM"]], "z_placeholder")


########################################################################################################################
########################################### DISCRIMINATORS #############################################################
########################################################################################################################

class Discriminator(Layer):

    def __init__(self, experiment, *args, **kwargs):
        super().__init__(experiment, *args, **kwargs)
        """
        init here all the layers you will need
        :param experiment: the experiment instance containing all the configuration parameters
        """
        self.X = tf.placeholder(tf.float32, [None] + experiment["SHAPE"], "X_placeholder")


########################################################################################################################
############################################## LOSSES ##################################################################
########################################################################################################################

class Loss(Layer):

    def __init__(self, experiment, *args, weight=1.0, **kwargs):
        super().__init__(experiment, *args, **kwargs)
        """
        Loss may have a weight that will globally resize it's importance
        Losses should not resize the output loss theirselves but use the weight value to eventually
        check that if it is 0, the loss may be not computed using then less resources.
        """
        self._weight = weight



########################################################################################################################
############################################### SOLVER #################################################################
########################################################################################################################

class Solver(Layer):
    """ Also called optimizer. """

    def __call__(self, params, **graph_nodes):
        self._pre_processing(**graph_nodes)
        res = self._forward(params, **graph_nodes)

        return res


########################################################################################################################
############################################### STATISTICS #############################################################
########################################################################################################################

class Statistic(Layer):

    def __init__(self, experiment, *args, base_path="", **kwargs):
        super().__init__(experiment, *args, **kwargs)
        self.base_path = base_path

    def __call__(self, **graph_nodes):
        self._pre_processing(**graph_nodes)
        return self._forward(**graph_nodes)
