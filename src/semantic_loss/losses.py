import os
import json
import tensorflow as tf
from functools import reduce

from base_layers import Loss
from thirdparties.py3psdd import Vtree, SddManager, PSddManager, io

from semantic_loss.util import _get_constraints_names
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori

EPSILON = 1e-9
"""
Module containing semantic losses, the base class pretty much already encapsulates everything needed, to add more
losses simply add the relative vtree and sdd files in their respective directories in "in/semantic_loss_constraints/",
the 2 files must have the same name, apart from ending in ".vtree" and ".sdd" respectively.
classes will be created automatically.
"""


class _SemanticLoss(Loss):
    """
    The forward method is mostly copied by
    https://github.com/UCLA-StarAI/Semantic-Loss/blob/master/complex_constraints/compute_mpe.py,
    from the semantic loss paper, currently it's basically the same class except for names changed to my liking
    and different imports and comments.
    """

    def _pre_processing(self, **graph_nodes):
        assert "G_output_logit" in graph_nodes, "Expected to find G_output_logit in graph nodes, which is the raw " \
                                                "output of the Generator (This loss will take care of applying the " \
                                                "sigmoid function)"

    def __init__(self, experiment, *args, **kwargs):
        super().__init__(experiment, *args, **kwargs)

        self.logger = self.experiment["LOGGER"]

        # if the input is probabilities already don't use sigmoid on it, otherwise use sigmoid
        self._use_sigmoid = not self.experiment["SEMANTIC_LOSS_INPUT_IS_PROBABILITIES"]

        self.constraint_name = self.__class__.__name__
        assert self.constraint_name[:13] == "SemanticLoss_", "Expected Semantic_ as prefix of the name of a" \
            " class extending SemanticLoss, the name is instead %s" % self.constraint_name
        self.constraint_name = "_".join(self.__class__.__name__.split("_")[1:])

        semantic_loss_from = experiment["SEMANTIC_LOSS_FROM_EPOCH"]
        self.semantic_loss_from = semantic_loss_from if semantic_loss_from is not None else 0
        self.semantic_loss_incremental = bool(experiment["SEMANTIC_LOSS_INCREMENTAL"])

    def _get_current_epoch(self, graph_nodes):
        """
        Get the current epoch as a tf node, automatically updates
        if G_global_step is used by any optimizer.

        :param graph_nodes:
        :return:
        """
        return graph_nodes["current_epoch"]

    def _incremental_timing(self, graph_nodes, semantic_loss):
        """
        If experiment["SEMANTIC_LOSS_INCREMENTAL"] is true then the semantic loss is adjusted with a weight
        equal to (current epoch + 1)/(total epochs + 1), otherwise the semantic loss is returned as it is.

        :param graph_nodes:
        :param semantic_loss:
        :return:
        """
        if self.semantic_loss_incremental:
            self.logger.info("Using incremental semantic loss")
            weight = tf.cast(
                self._get_current_epoch(graph_nodes) + 1 - self.semantic_loss_from,
                tf.float32
            ) / tf.cast(self.experiment["LEARNING_EPOCHS"] + 1 - self.semantic_loss_from, tf.float32)
            return semantic_loss * weight
        else:
            self.logger.info("Not using incremental semantic loss")
            return semantic_loss

    def _time_semantic_loss(self, graph_nodes, semantic_loss):
        """
        Get the semantic loss as a tf node which value is zero if the current epoch is lower
        than the experiment["SEMANTIC_LOSS_FROM_EPOCH"] parameter, otherwise the semantic loss is returned.

        :param graph_nodes:
        :param semantic_loss:
        :return:
        """
        self.logger.info("Using semantic loss starting from epoch %s" % self.semantic_loss_from)
        cond = self._get_current_epoch(graph_nodes) >= self.semantic_loss_from
        truefn = lambda: semantic_loss
        falsefn = lambda: tf.constant(0.)
        epoch_timed = tf.cond(cond, true_fn=truefn, false_fn=falsefn)
        incremental_timed = self._incremental_timing(graph_nodes, epoch_timed)
        return incremental_timed

    @staticmethod
    def _import_psdd(constraint_name):
        """
        Given a constraint_name, assert the existence and look for the related .vtree and .sdd files, which
        are expected to be in in/semantic_loss_constraints/constraints_as_vtree and /constraints_as_sdd
        respectively, and to be named as constraint_name.vtree and .sdd respectively.
        The vtree and sdd are loaded and used to instantiate the psdd, which is then returned.
        :param constraint_name: Name of the constraints to use.
        """
        cwd = os.getcwd()
        assert cwd[-4:] == "/src", "Expected to be in the src directory of the repository, the cwd is instead %s" % cwd

        constraint_name = constraint_name.replace("SemanticLoss_", "")
        vtree = "in/semantic_loss_constraints/constraints_as_vtree/" + constraint_name + ".vtree"
        sdd = "in/semantic_loss_constraints/constraints_as_sdd/" + constraint_name + ".sdd"
        assert os.path.isfile(vtree), vtree + " is not a file."
        assert os.path.isfile(sdd), sdd + " is not a file."

        # load vtree and sdd files and construct the PSDD
        vtree = Vtree.read(vtree)
        manager = SddManager(vtree)
        alpha = io.sdd_read(sdd, manager)
        pmanager = PSddManager(vtree)
        psdd = pmanager.copy_and_normalize_sdd(alpha, vtree)

        return psdd

    def _forward(self, **graph_nodes):
        """
        Returns the semantic loss related to the instance of this class, using the G_output_logit node.
        The sigmoid function is applied by the loss if the property use_sigmoid is set to True.

        :param graph_nodes: Dict of tf graph nodes, G_output_logit must be present.
        :return: A pair of dicts, where the first one is for giving nodes to the trainer, the second is for providing
        nodes that we want to be logged.
        The first dict maps:
        - "G_loss" to the semantic loss of this class
        - self.__class__.name to the semantic loss of this class
        - self.__class__.name + "_wmc" to the wmc defined by the vtree and sdd for this class
        The second dict maps:
        - "G_loss" to the semantic loss of this class
        - self.__class__.name to the semantic loss of this class
        - self.__class__.name + "_wmc" to the wmc defined by the vtree and sdd for this class
        - self.__class__.name + "_wmc_per_sample" to the wmc defined by the vtree and sdd for this class, over each
            sample.
        """
        self.logger.info("Semantic loss %s " % self.constraint_name)
        self.logger.info("Importing psdd %s " % self.constraint_name)
        psdd = self.__class__._import_psdd(self.constraint_name)
        g_output_logit = graph_nodes["G_output_logit"]
        self.logger.info("Semantic loss found G_output_logit of shape %s" % g_output_logit.shape)

        # set values to probabilities
        toprobs = g_output_logit
        if self.use_sigmoid:
            if "SHAPE" in self.experiment and self.experiment["SHAPE"][-1] > 2:
                toprobs = tf.nn.softmax(g_output_logit, name=self.constraint_name + "_to_probabilities")
            else:
                toprobs = tf.nn.sigmoid(g_output_logit, name=self.constraint_name + "_to_probabilities")

        # need to reshape as a 1d vector of variables for each sample, needed by psdd for the tf AC
        total_variables = reduce(lambda x, y: x * y, toprobs.shape[1:])
        probs_as_vector = tf.reshape(toprobs, [self.experiment["BATCH_SIZE"], total_variables],
                                     name=self.constraint_name + "_reshape_to_vector")
        self.logger.info("Semantic loss reshaping G_output_logit to %s" % probs_as_vector.shape)

        nodes = dict()
        wmc_per_sample = psdd.generate_tf_ac_v2(probs_as_vector, self.experiment["BATCH_SIZE"])
        self.logger.info("Semantic loss wmc of shape %s" % wmc_per_sample.shape)
        wmc = tf.reduce_mean(wmc_per_sample)
        self.logger.info("Semantic loss reduced wmc of shape %s" % wmc.shape)
        semantic_loss_pre_timing = -tf.log(wmc)
        semantic_loss = self._time_semantic_loss(graph_nodes, semantic_loss_pre_timing)

        nodes["G_loss"] = semantic_loss  # needed cos its expected by the trainer
        nodes[self.constraint_name] = semantic_loss_pre_timing
        nodes[self.constraint_name + "_wmc"] = wmc
        nodes[self.constraint_name + "_wmc_per_sample"] = wmc_per_sample
        nodes_to_log = {**nodes}
        nodes_to_log.pop(self.constraint_name + "_wmc_per_sample")

        del psdd
        return nodes, nodes_to_log, dict()

    @property
    def use_sigmoid(self):
        """
        Is sigmoid being applied to G_output_logit?
        :return:
        """
        return self._use_sigmoid

    @use_sigmoid.setter
    def use_sigmoid(self, value):
        """
        Set if sigmoid should be applied to G_output_logit.
        :return:
        """
        assert (isinstance(value, bool)), "function use_sigmoid expects a boolean value"
        self._use_sigmoid = value


class SemanticLoss_pc_custom(_SemanticLoss):

    def _forward(self, **graph_nodes):
        """
        Returns the semantic loss related to the instance of this class, using the G_output_logit node.
        The sigmoid function is applied by the loss if the property use_sigmoid is set to True.

        :param graph_nodes: Dict of tf graph nodes, G_output_logit must be present.
        :return: A pair of dicts, where the first one is for giving nodes to the trainer, the second is for providing
        nodes that we want to be logged.
        The first dict maps:
        - "G_loss" to the semantic loss of this class
        - self.__class__.name to the semantic loss of this class
        - self.__class__.name + "_wmc" to the wmc defined by the vtree and sdd for this class
        The second dict maps:
        - "G_loss" to the semantic loss of this class
        - self.__class__.name to the semantic loss of this class
        - self.__class__.name + "_wmc" to the wmc defined by the vtree and sdd for this class
        - self.__class__.name + "_wmc_per_sample" to the wmc defined by the vtree and sdd for this class, over each
            sample.
        """
        self.logger.info("Semantic loss %s " % self.constraint_name)
        self.logger.info("Importing psdd %s " % self.constraint_name)
        psdd = self.__class__._import_psdd(self.constraint_name)
        g_output_logit = graph_nodes["G_output_logit"]
        self.logger.info("Semantic loss found G_output_logit of shape %s" % g_output_logit.shape)

        # set values to probabilities
        toprobs = g_output_logit
        if self.use_sigmoid:
            toprobs = tf.nn.sigmoid(g_output_logit, name=self.constraint_name + "_to_probabilities")

        """
        Get rows and columns, stack them as a tensor of shape [batch_size * 18, 10], the 18 comes from
        the fact that we will be checking 9 rows and 8 cols for every sample, 10 because we are checking for each
        rows/cols the constraints on 10 variables.
        """
        rows = toprobs[:, 1:10, 0:10, :]
        cols = toprobs[:, 0:10, 1:10, :]
        cols = tf.transpose(cols, [0, 2, 1, 3])
        aggregate = tf.concat([rows, cols], axis=1)
        aggregate = tf.reshape(aggregate, (-1, 10))
        self.logger.info("Semantic loss reshaping variables to shape %s" % aggregate.shape)

        """
        Once we have the aggregate [batch_size * 18, 10] tensor pass it thought the circuit,
        reduce_prod on constraints related to different rows/cols of the same data sample (as if we were doing
        an AND), then reduce mean over the samples.
        """
        wmc = psdd.generate_tf_ac_v2(aggregate)
        self.logger.info("Semantic loss wmc of shape %s" % wmc.shape)
        wmc = tf.reshape(wmc, [-1, 18])
        self.logger.info("Semantic loss wmc reshaped to shape %s" % wmc.shape)
        wmc_per_sample = tf.reduce_prod(wmc, axis=1)
        self.logger.info("Semantic loss wmc reduce product to shape %s" % wmc_per_sample.shape)
        wmc = tf.reduce_mean(wmc_per_sample)
        self.logger.info("Semantic loss wmc reduced mean to shape %s" % wmc.shape)
        semantic_loss_pre_timing = -tf.log(wmc)
        semantic_loss = self._time_semantic_loss(graph_nodes, semantic_loss_pre_timing)

        # fill dict and return
        nodes = dict()
        nodes["G_loss"] = semantic_loss  # needed cos its expected by the trainer
        # add these nodes so that statistics can pick them up later if needed
        nodes["SemanticLoss_" + self.constraint_name] = semantic_loss_pre_timing
        nodes["SemanticLoss_" + self.constraint_name + "_wmc"] = wmc
        nodes["SemanticLoss_" + self.constraint_name + "_wmc_per_sample"] = wmc_per_sample
        nodes_to_log = {**nodes}
        nodes_to_log.pop("SemanticLoss_" + self.constraint_name + "_wmc_per_sample")

        del psdd

        return nodes, nodes_to_log, dict()


class SemanticLoss_pc_custom_full(_SemanticLoss):
    
    def __init__(self, experiment, *args, **kwargs):
        super().__init__(experiment, *args, **kwargs)

        self.constraint_name = "pc_custom"

    def _forward(self, **graph_nodes):
        """
        Returns the semantic loss related to the instance of this class, using the G_output_logit node.
        The sigmoid function is applied by the loss if the property use_sigmoid is set to True.

        :param graph_nodes: Dict of tf graph nodes, G_output_logit must be present.
        :return: A pair of dicts, where the first one is for giving nodes to the trainer, the second is for providing
        nodes that we want to be logged.
        The first dict maps:
        - "G_loss" to the semantic loss of this class
        - self.__class__.name to the semantic loss of this class
        - self.__class__.name + "_wmc" to the wmc defined by the vtree and sdd for this class
        The second dict maps:
        - "G_loss" to the semantic loss of this class
        - self.__class__.name to the semantic loss of this class
        - self.__class__.name + "_wmc" to the wmc defined by the vtree and sdd for this class
        - self.__class__.name + "_wmc_per_sample" to the wmc defined by the vtree and sdd for this class, over each
            sample.
        """
        self.logger.info("Semantic loss %s " % self.constraint_name)
        self.logger.info("Importing psdd %s " % self.constraint_name)
        psdd = self.__class__._import_psdd(self.constraint_name)
        g_output_logit = graph_nodes["G_output_logit"]
        self.logger.info("Semantic loss found G_output_logit of shape %s" % g_output_logit.shape)

        # set values to probabilities
        toprobs = g_output_logit
        if self.use_sigmoid:
            toprobs = tf.nn.sigmoid(g_output_logit, name=self.constraint_name + "_to_probabilities")

        """
        Get rows and columns, stack them as a tensor of shape [batch_size * 18, 10], the 18 comes from
        the fact that we will be checking 9 rows and 8 cols for every sample, 10 because we are checking for each
        rows/cols the constraints on 10 variables.
        """
        rows_left = toprobs[:, 1:19, 0:10, :]
        rows_right = toprobs[:, 1:19, 10:20]
        rows_right = rows_right[:, :, ::-1, :]
        cols_top = toprobs[:, 0:10, 1:19, :]
        cols_top = tf.transpose(cols_top, [0, 2, 1, 3])
        cols_bottom = toprobs[:, 10:20, 1:19, :]
        cols_bottom = cols_bottom[:, ::-1, :, :]
        cols_bottom = tf.transpose(cols_bottom, [0, 2, 1, 3])
        aggregate = tf.concat([rows_left, rows_right, cols_top, cols_bottom], axis=1)
        aggregate = tf.reshape(aggregate, (-1, 10))
        self.logger.info("Semantic loss reshaping variables to shape %s" % aggregate.shape)

        """
        Once we have the aggregate [batch_size * 18, 10] tensor pass it thought the circuit,
        reduce_prod on constraints related to different rows/cols of the same data sample (as if we were doing
        an AND), then reduce mean over the samples.
        """
        wmc = psdd.generate_tf_ac_v2(aggregate)
        self.logger.info("Semantic loss wmc of shape %s" % wmc.shape)
        wmc = tf.reshape(wmc, [-1, 18 * 4])
        self.logger.info("Semantic loss wmc reshaped to shape %s" % wmc.shape)
        wmc_per_sample = tf.reduce_prod(wmc, axis=1)
        self.logger.info("Semantic loss wmc reduce product to shape %s" % wmc_per_sample.shape)
        wmc = tf.reduce_mean(wmc_per_sample)
        self.logger.info("Semantic loss wmc reduced mean to shape %s" % wmc.shape)
        semantic_loss_pre_timing = -tf.log(wmc)
        semantic_loss = self._time_semantic_loss(graph_nodes, semantic_loss_pre_timing)

        # fill dict and return
        nodes = dict()
        nodes["G_loss"] = semantic_loss  # needed cos its expected by the trainer
        # add these nodes so that statistics can pick them up later if needed
        nodes["SemanticLoss_" + self.constraint_name + "_full"] = semantic_loss_pre_timing
        nodes["SemanticLoss_" + self.constraint_name + "_full_wmc"] = wmc
        nodes["SemanticLoss_" + self.constraint_name + "_full_wmc_per_sample"] = wmc_per_sample
        nodes_to_log = {**nodes}
        nodes_to_log.pop("SemanticLoss_" + self.constraint_name + "_full_wmc_per_sample")

        del psdd

        return nodes, nodes_to_log, dict()


class SemanticLoss_pc_custom_rows(_SemanticLoss):
    
    def __init__(self, experiment, *args, **kwargs):
        super().__init__(experiment, *args, **kwargs)

        self.constraint_name = "pc_custom"

    def _forward(self, **graph_nodes):
        """
        Returns the semantic loss related to the instance of this class, using the G_output_logit node.
        The sigmoid function is applied by the loss if the property use_sigmoid is set to True.

        :param graph_nodes: Dict of tf graph nodes, G_output_logit must be present.
        :return: A pair of dicts, where the first one is for giving nodes to the trainer, the second is for providing
        nodes that we want to be logged.
        The first dict maps:
        - "G_loss" to the semantic loss of this class
        - self.__class__.name to the semantic loss of this class
        - self.__class__.name + "_wmc" to the wmc defined by the vtree and sdd for this class
        The second dict maps:
        - "G_loss" to the semantic loss of this class
        - self.__class__.name to the semantic loss of this class
        - self.__class__.name + "_wmc" to the wmc defined by the vtree and sdd for this class
        - self.__class__.name + "_wmc_per_sample" to the wmc defined by the vtree and sdd for this class, over each
            sample.
        """
        self.logger.info("Semantic loss %s " % self.constraint_name)
        self.logger.info("Importing psdd %s " % self.constraint_name)
        psdd = self.__class__._import_psdd(self.constraint_name)
        g_output_logit = graph_nodes["G_output_logit"]
        self.logger.info("Semantic loss found G_output_logit of shape %s" % g_output_logit.shape)

        # set values to probabilities
        toprobs = g_output_logit
        if self.use_sigmoid:
            toprobs = tf.nn.sigmoid(g_output_logit, name=self.constraint_name + "_to_probabilities")

        """
        Get rows and columns, stack them as a tensor of shape [batch_size * 18, 10], the 18 comes from
        the fact that we will be checking 9 rows and 8 cols for every sample, 10 because we are checking for each
        rows/cols the constraints on 10 variables.
        """
        rows_left = toprobs[:, 1:19, 0:10, :]
        rows_right = toprobs[:, 1:19, 10:20]
        rows_right = rows_right[:, :, ::-1, :]
        aggregate = tf.concat([rows_left, rows_right], axis=1)
        aggregate = tf.reshape(aggregate, (-1, 10))
        self.logger.info("Semantic loss reshaping variables to shape %s" % aggregate.shape)

        """
        Once we have the aggregate [batch_size * 18, 10] tensor pass it thought the circuit,
        reduce_prod on constraints related to different rows/cols of the same data sample (as if we were doing
        an AND), then reduce mean over the samples.
        """
        wmc = psdd.generate_tf_ac_v2(aggregate)
        self.logger.info("Semantic loss wmc of shape %s" % wmc.shape)
        wmc = tf.reshape(wmc, [-1, 18 * 2])
        self.logger.info("Semantic loss wmc reshaped to shape %s" % wmc.shape)
        wmc_per_sample = tf.reduce_prod(wmc, axis=1)
        self.logger.info("Semantic loss wmc reduce product to shape %s" % wmc_per_sample.shape)
        wmc = tf.reduce_mean(wmc_per_sample)
        self.logger.info("Semantic loss wmc reduced mean to shape %s" % wmc.shape)
        semantic_loss_pre_timing = -tf.log(wmc)
        semantic_loss = self._time_semantic_loss(graph_nodes, semantic_loss_pre_timing)

        # fill dict and return
        nodes = dict()
        nodes["G_loss"] = semantic_loss  # needed cos its expected by the trainer
        # add these nodes so that statistics can pick them up later if needed
        nodes["SemanticLoss_" + self.constraint_name + "_rows"] = semantic_loss_pre_timing
        nodes["SemanticLoss_" + self.constraint_name + "_rows_wmc"] = wmc
        nodes["SemanticLoss_" + self.constraint_name + "_rows_wmc_per_sample"] = wmc_per_sample
        nodes_to_log = {**nodes}
        nodes_to_log.pop("SemanticLoss_" + self.constraint_name + "_rows_wmc_per_sample")

        del psdd

        return nodes, nodes_to_log, dict()


class SemanticLoss_mnist_pc_custom_full(_SemanticLoss):
    
    def __init__(self, experiment, *args, **kwargs):
        super().__init__(experiment, *args, **kwargs)

        self.constraint_name = "mnist_pc_custom"

    def _forward(self, **graph_nodes):
        """
        Returns the semantic loss related to the instance of this class, using the G_output_logit node.
        The sigmoid function is applied by the loss if the property use_sigmoid is set to True.

        :param graph_nodes: Dict of tf graph nodes, G_output_logit must be present.
        :return: A pair of dicts, where the first one is for giving nodes to the trainer, the second is for providing
        nodes that we want to be logged.
        The first dict maps:
        - "G_loss" to the semantic loss of this class
        - self.__class__.name to the semantic loss of this class
        - self.__class__.name + "_wmc" to the wmc defined by the vtree and sdd for this class
        The second dict maps:
        - "G_loss" to the semantic loss of this class
        - self.__class__.name to the semantic loss of this class
        - self.__class__.name + "_wmc" to the wmc defined by the vtree and sdd for this class
        - self.__class__.name + "_wmc_per_sample" to the wmc defined by the vtree and sdd for this class, over each
            sample.
        """
        self.logger.info("Semantic loss %s " % self.constraint_name)
        self.logger.info("Importing psdd %s " % self.constraint_name)
        psdd = self.__class__._import_psdd(self.constraint_name)
        g_output_logit = graph_nodes["G_output_logit"]
        self.logger.info("Semantic loss found G_output_logit of shape %s" % g_output_logit.shape)

        # set values to probabilities
        toprobs = g_output_logit
        if self.use_sigmoid:
            toprobs = tf.nn.sigmoid(g_output_logit, name=self.constraint_name + "_to_probabilities")

        """
        Get rows and columns, stack them as a tensor of shape [batch_size * 18, 10], the 18 comes from
        the fact that we will be checking 9 rows and 8 cols for every sample, 10 because we are checking for each
        rows/cols the constraints on 10 variables.
        """
        rows_left = toprobs[:, 1:27, 0:14, :]
        rows_right = toprobs[:, 1:27, 14:28]
        rows_right = rows_right[:, :, ::-1, :]
        aggregate = tf.concat([rows_left, rows_right], axis=1)
        aggregate = tf.reshape(aggregate, (-1, 14))
        self.logger.info("Semantic loss reshaping variables to shape %s" % aggregate.shape)

        """
        Once we have the aggregate [batch_size * 18, 10] tensor pass it thought the circuit,
        reduce_prod on constraints related to different rows/cols of the same data sample (as if we were doing
        an AND), then reduce mean over the samples.
        """
        wmc = psdd.generate_tf_ac_v2(aggregate)
        self.logger.info("Semantic loss wmc of shape %s" % wmc.shape)
        wmc = tf.reshape(wmc, [-1, 26 * 2])
        self.logger.info("Semantic loss wmc reshaped to shape %s" % wmc.shape)
        wmc_per_sample = tf.reduce_prod(wmc, axis=1)
        self.logger.info("Semantic loss wmc reduce product to shape %s" % wmc_per_sample.shape)
        wmc = tf.reduce_mean(wmc_per_sample)
        self.logger.info("Semantic loss wmc reduced mean to shape %s" % wmc.shape)
        semantic_loss_pre_timing = -tf.log(wmc)
        semantic_loss = self._time_semantic_loss(graph_nodes, semantic_loss_pre_timing)

        # fill dict and return
        nodes = dict()
        nodes["G_loss"] = semantic_loss  # needed cos its expected by the trainer
        # add these nodes so that statistics can pick them up later if needed
        nodes["SemanticLoss_" + self.constraint_name + "_full"] = semantic_loss_pre_timing
        nodes["SemanticLoss_" + self.constraint_name + "_full_wmc"] = wmc
        nodes["SemanticLoss_" + self.constraint_name + "_full_wmc_per_sample"] = wmc_per_sample
        nodes_to_log = {**nodes}
        nodes_to_log.pop("SemanticLoss_" + self.constraint_name + "_full_wmc_per_sample")

        del psdd

        return nodes, nodes_to_log, dict()


class SemanticLoss_imply_feature(_SemanticLoss):
    """
    Gets some frequent itemsets and uses their presence (or lack of) in fake and real samples
    as features/constraints.
    """

    def __init__(self, experiment, *args, **kwargs):
        super().__init__(experiment, *args, **kwargs)

        experiment["LOGGER"].info("Computing frequent itemsets")
        data = experiment.training_data
        data = np.reshape(data, [data.shape[0], -1])
        df = pd.DataFrame(data, dtype=np.bool)
        sets = apriori(df, min_support=0.15, verbose=True)
        sets = sets[sets["itemsets"].map(len) >= 5]
        sets = sets["itemsets"].tolist()
        maxlen = max([len(s) for s in sets])
        indexes = []
        for s in sets:
            s = list(s)
            while len(s) != maxlen:
                s.append(s[0])
            indexes.append(s)
        self.indexes = indexes
        self.maxlen = maxlen

    def _indexes_to_gather(self, bs):
        """
        Bump each index in indexes by the total number of Dz variables, and prepend
        to each group of indexes (itemsets) the index of the variable in Dz it
        is related to.
        :return:
        """
        indexes = [[i] + [index + len(self.indexes) for index in index_set] for i, index_set in enumerate(self.indexes)]
        indexes = [index for index_set in indexes for index in index_set]
        indexes = [[i, index] for i in range(bs) for index in indexes]
        return indexes

    def _forward(self, **graph_nodes):
        psdd = self.__class__._import_psdd(self.constraint_name)
        g_output_logit = graph_nodes["G_output_logit"]

        # set values to probabilities
        toprobs = g_output_logit
        if self.use_sigmoid:
            toprobs = tf.nn.sigmoid(g_output_logit, name=self.constraint_name + "_to_probabilities")

        bs = self.experiment["BATCH_SIZE"]
        total_variables = reduce(lambda x, y: x * y, toprobs.shape[1:])
        toprobs = tf.reshape(toprobs, [bs, total_variables])

        # discretize noise, which implies features
        Dz = graph_nodes["Dz"]
        Dz = tf.reshape(Dz, [bs, 377])
        vars = tf.concat([Dz, toprobs], axis=1)
        indexes_to_gather = self._indexes_to_gather(bs)
        vars = tf.gather_nd(vars, indexes_to_gather)
        vars = tf.reshape(vars, [-1, (self.maxlen + 1)])

        wmc = psdd.generate_tf_ac_v2(vars)
        wmc = tf.reshape(wmc, [bs, len(self.indexes)])
        wmc = tf.reduce_prod(wmc, axis=1)
        wmc = tf.reduce_mean(wmc)
        semantic_loss_pre_timing = -tf.log(wmc)
        semantic_loss = self._time_semantic_loss(graph_nodes, semantic_loss_pre_timing)

        # fill dict and return
        nodes = dict()
        nodes["G_loss"] = semantic_loss  # needed cos its expected by the trainer
        # add these nodes so that statistics can pick them up later if needed
        nodes["SemanticLoss_" + self.constraint_name + "_full"] = semantic_loss_pre_timing
        nodes["SemanticLoss_" + self.constraint_name + "_full_wmc"] = wmc

        del psdd

        return nodes, nodes, dict()


class SemanticLoss_still_life(_SemanticLoss):
   
    def __init__(self, experiment, *args, **kwargs):
        super().__init__(experiment, *args, **kwargs)

    def convert_index(self, row, col):
        return row * self.experiment["SHAPE"][0] + col

    def corner_indexes(self):
        """
        Return the indexes to gather for each corner
        :return:
        """
        corner_variables = []
        bs = self.experiment["BATCH_SIZE"]
        shape = self.experiment["SHAPE"]
        rows = shape[0]
        cols = shape[1]

        # top left corner
        corner_variables.append(self.convert_index(0, 0))
        corner_variables.append(self.convert_index(0, 1))
        corner_variables.append(self.convert_index(1, 0))
        corner_variables.append(self.convert_index(1, 1))

        # bottom left
        corner_variables.append(self.convert_index(rows - 1, 0))
        corner_variables.append(self.convert_index(rows - 1, 1))
        corner_variables.append(self.convert_index(rows - 2, 0))
        corner_variables.append(self.convert_index(rows - 2, 1))

        # top right
        corner_variables.append(self.convert_index(0, cols - 1))
        corner_variables.append(self.convert_index(0, cols - 2))
        corner_variables.append(self.convert_index(1, cols - 1))
        corner_variables.append(self.convert_index(1, cols - 2))

        # bottom right
        corner_variables.append(self.convert_index(rows - 1, cols - 1))
        corner_variables.append(self.convert_index(rows - 1, cols - 2))
        corner_variables.append(self.convert_index(rows - 2, cols - 1))
        corner_variables.append(self.convert_index(rows - 2, cols - 2))

        indexes = [[b, var] for b in range(bs) for var in corner_variables]

        # 4 corners * bs * 4 variables per corner
        assert len(indexes) == (4 * bs * 4)
        for index in indexes:
            sample, var_index = index
            assert var_index < (rows * cols)
            assert var_index >= 0

        return indexes

    def edge_indexes(self):
        """
        Return the indexes to gather for each edge
        :return:
        """
        edge_variables = []
        bs = self.experiment["BATCH_SIZE"]
        shape = self.experiment["SHAPE"]
        rows = shape[0]
        cols = shape[1]

        # left edge
        relative = [[0, 0], [-1, 0], [1, 0], [0, 1], [-1, 1], [1, 1]]
        for row in range(1, rows - 1):
            indexes = [[row + pair[0], 0 + pair[1]] for pair in relative]
            indexes = [self.convert_index(pair[0], pair[1]) for pair in indexes]
            edge_variables.extend(indexes)

        # right edge
        relative = [[0, 0], [-1, 0], [1, 0], [0, -1], [-1, -1], [1, -1]]
        for row in range(1, rows - 1):
            indexes = [[row + pair[0], cols - 1 + pair[1]] for pair in relative]
            indexes = [self.convert_index(pair[0], pair[1]) for pair in indexes]
            edge_variables.extend(indexes)

        # top edge
        relative = [[0, 0], [0, 1], [0, -1], [1, 0], [1, -1], [1, 1]]
        for col in range(1, cols - 1):
            indexes = [[0 + pair[0], col + pair[1]] for pair in relative]
            indexes = [self.convert_index(pair[0], pair[1]) for pair in indexes]
            edge_variables.extend(indexes)

        # bottom edge
        relative = [[0, 0], [0, 1], [0, -1], [-1, 0], [-1, -1], [-1, 1]]
        for col in range(1, cols - 1):
            indexes = [[rows - 1 + pair[0], col + pair[1]] for pair in relative]
            indexes = [self.convert_index(pair[0], pair[1]) for pair in indexes]
            edge_variables.extend(indexes)

        indexes = [[b, var] for b in range(bs) for var in edge_variables]

        # bs * total edge variables * total variables per edge variable
        assert len(indexes) == (bs * (2 * (rows - 2) + 2 * (cols - 2)) * 6)
        for index in indexes:
            sample, var_index = index
            assert var_index < (rows * cols)
            assert var_index >= 0

        return indexes

    def internal_indexes(self):
        """
        Return the indexes to gather for each edge
        :return:
        """
        variables = []
        bs = self.experiment["BATCH_SIZE"]
        shape = self.experiment["SHAPE"]
        rows = shape[0]
        cols = shape[1]

        # for each internal variable add 9 indexes (self + neighbours)
        relative = [[0, 0], [-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]]
        for r in range(1, rows - 1):
            for c in range(1, cols - 1):
                for rpair in relative:
                    variables.append(self.convert_index(r + rpair[0], c + rpair[1]))

        indexes = [[b, var] for b in range(bs) for var in variables]

        # bs * internal variables * variable for each internal variable (neighbours + self)
        assert len(indexes) == (bs * (rows - 2) * (cols - 2) * 9)
        for index in indexes:
            sample, var_index = index
            assert var_index < (rows * cols)
            assert var_index >= 0
        return indexes

    def _forward(self, **graph_nodes):
        """
        Returns the semantic loss related to the instance of this class, using the G_output_logit node.
        The sigmoid function is applied by the loss if the property use_sigmoid is set to True.

        :param graph_nodes: Dict of tf graph nodes, G_output_logit must be present.
        :return: A pair of dicts, where the first one is for giving nodes to the trainer, the second is for providing
        nodes that we want to be logged.
        The first dict maps:
        - "G_loss" to the semantic loss of this class
        - self.__class__.name to the semantic loss of this class
        - self.__class__.name + "_wmc" to the wmc defined by the vtree and sdd for this class
        The second dict maps:
        - "G_loss" to the semantic loss of this class
        - self.__class__.name to the semantic loss of this class
        - self.__class__.name + "_wmc" to the wmc defined by the vtree and sdd for this class
        - self.__class__.name + "_wmc_per_sample" to the wmc defined by the vtree and sdd for this class, over each
            sample.
        """
        self.logger.info("Semantic loss %s " % self.constraint_name)
        g_output_logit = graph_nodes["G_output_logit"]
        self.logger.info("Semantic loss found G_output_logit of shape %s" % g_output_logit.shape)

        # set values to probabilities
        toprobs = g_output_logit
        if self.use_sigmoid:
            toprobs = tf.nn.sigmoid(g_output_logit, name=self.constraint_name + "_to_probabilities")

        bs = self.experiment["BATCH_SIZE"]
        total_variables = reduce(lambda x, y: x * y, toprobs.shape[1:])
        toprobs = tf.reshape(toprobs, [bs, total_variables])
        self.logger.info("Semantic loss reshaping toprobs to %s" % toprobs.shape)
        shape = self.experiment["SHAPE"]
        rows = shape[0]
        cols = shape[1]

        # get corner variables
        corner_indexes = self.corner_indexes()
        corner_vars = tf.gather_nd(toprobs, corner_indexes)
        self.logger.info("Semantic loss gathered corner variables shape %s" % corner_vars.shape)
        # reshape as [bs * # corners, variables per corner]
        corner_vars = tf.reshape(corner_vars, [bs * 4, 4])
        self.logger.info("Semantic loss gathered corner variables reshaped to %s" % corner_vars.shape)

        # get edge variables
        edge_indexes = self.edge_indexes()
        edge_vars = tf.gather_nd(toprobs, edge_indexes)
        self.logger.info("Semantic loss gathered edge  variables shape %s" % edge_vars.shape)
        # reshape as [bs * edge variables, variables per edge variable]
        edge_vars = tf.reshape(edge_vars, [bs * ((rows - 2) * 2 + ((cols - 2) * 2)), 6])
        self.logger.info("Semantic loss gathered edge variables reshaped to %s" % edge_vars.shape)

        # get internal variables
        internal_indexes = self.internal_indexes()
        internal_vars = tf.gather_nd(toprobs, internal_indexes)
        self.logger.info("Semantic loss gathered internal  variables shape %s" % internal_vars.shape)
        # reshape as [bs * internal variables, variables per internal variable]
        internal_vars = tf.reshape(internal_vars, [bs * (rows - 2) * (cols - 2), 9])
        self.logger.info("Semantic loss gathered internal variables reshaped to %s" % internal_vars.shape)

        """
        For each type of group of variables run it through the related
        arithmetic circuit.
        """
        self.logger.info("Importing psdd still_life_corners")
        corner_psdd = self.__class__._import_psdd("still_life_corners")
        corner_wmc = corner_psdd.generate_tf_ac_v2(corner_vars)
        self.logger.info("Semantic loss corner wmc shape %s" % corner_wmc.shape)
        # reshape to [bs, number of corners]
        corner_wmc = tf.reshape(corner_wmc, [bs, 4])
        self.logger.info("Semantic loss corner wmc reshaped to %s" % corner_wmc.shape)
        del corner_psdd

        self.logger.info("Importing psdd still_life_edges")
        edge_psdd = self.__class__._import_psdd("still_life_edges")
        edge_wmc = edge_psdd.generate_tf_ac_v2(edge_vars)
        self.logger.info("Semantic loss edge wmc shape %s" % edge_wmc.shape)
        # reshape to [bs, number of edge variables]
        edge_wmc = tf.reshape(edge_wmc, [bs, ((rows - 2) * 2 + (cols - 2) * 2)])
        self.logger.info("Semantic loss edge wmc reshaped to %s" % edge_wmc.shape)
        del edge_psdd

        self.logger.info("Importing psdd still_life_internals")
        internal_psdd = self.__class__._import_psdd("still_life_internals")
        internal_wmc = internal_psdd.generate_tf_ac_v2(internal_vars)
        self.logger.info("Semantic loss internal wmc shape %s" % internal_wmc.shape)
        # reshape to [bs, number of internal variables]
        internal_wmc = tf.reshape(internal_wmc, [bs, (rows - 2) * (cols - 2)])
        self.logger.info("Semantic loss internal wmc reshaped to %s" % internal_wmc.shape)
        del internal_psdd

        """
        Stuff related to the same batch sample should be on the same dimension, i.e. for sample i
        we find wmcs related to it in corner_wmc[i,...], edge_wmc[i,...] and internal_wmc[i,...], concat
        everything on axis1 then reduce prod to apply an "AND" to these constraints.
        """
        wmc = tf.concat([corner_wmc, edge_wmc, internal_wmc], axis=1)

        # mask out a certain rate of constraints to be "ANDED" to avoid gradient vanishing
        mask = tf.random_uniform(wmc.shape)
        thres = self.experiment["SEMANTIC_LOSS_CONSTRAINTS_DROPOUT_RATE"]
        if self.experiment["SEMANTIC_LOSS_CONSTRAINTS_DROPOUT_DECREMENTAL"]:
            self.logger.info("Using decrement over time for constraints dropout")
            weight = 1.0 - (tf.cast(self._get_current_epoch(graph_nodes) + 1, tf.float32) / (
                    self.experiment["LEARNING_EPOCHS"] + 1))
            thres = thres * weight

        mask1 = tf.cast((mask >= thres), tf.float32) * wmc
        mask0 = tf.cast((mask < thres), tf.float32) * tf.ones(wmc.shape)
        wmc = mask1 + mask0

        self.logger.info("Semantic loss concatenating on axis=1 corner, edge, internal wmcs, "
                         "obtained shape %s" % wmc.shape)
        wmc_per_sample = tf.reduce_prod(wmc, axis=1)
        self.logger.info("Semantic loss reduced product wmc shape %s" % wmc_per_sample.shape)
        wmc = tf.reduce_mean(wmc_per_sample)
        self.logger.info("Semantic loss reduced mean wmc shape %s" % wmc.shape)
        semantic_loss_pre_timing = -tf.log(wmc + 1e-8)
        if self.experiment["SL_EQUALIZE"]:
            alpha = tf.abs(tf.stop_gradient(graph_nodes["G_adversarial_loss"] / semantic_loss_pre_timing))
            semantic_loss_pre_timing *= alpha

        semantic_loss = self._time_semantic_loss(graph_nodes, semantic_loss_pre_timing)

        # fill dict and return
        nodes = dict()
        nodes["G_loss"] = semantic_loss  # needed cos its expected by the trainer
        # add these nodes so that statistics can pick them up later if needed
        nodes["SemanticLoss_" + self.constraint_name] = semantic_loss_pre_timing
        nodes["SemanticLoss_" + self.constraint_name + "_wmc"] = wmc
        nodes["SemanticLoss_" + self.constraint_name + "_wmc_per_sample"] = wmc_per_sample
        nodes_to_log = {**nodes}
        nodes_to_log.pop("SemanticLoss_" + self.constraint_name + "_wmc_per_sample")

        return nodes, nodes_to_log, dict()


class SemanticLoss_still_life_window(SemanticLoss_still_life):
    
    def __init__(self, experiment, *args, **kwargs):
        super().__init__(experiment, *args, **kwargs)

        self.wsize = self.experiment["STILL_LIFE_WINDOW_SIZE"]

    def window_indexes(self):
        """
        Return the indexes to gather for each window, expexts that you got your window size w.r.t
        total size right (does not perform correctness checks).
        :return:
        """
        variables = []
        bs = self.experiment["BATCH_SIZE"]
        shape = self.experiment["SHAPE"]
        rows = shape[0]
        cols = shape[1]

        # w_x and w_y are the x and y coordinates for the windows
        for w_x in range(0, rows, self.wsize - 2):
            for w_y in range(0, cols, self.wsize - 2):
                # not sure it generalizes~
                if (w_x + self.wsize) <= rows and (w_y + self.wsize) <= cols:
                    for x in range(self.wsize):
                        for y in range(self.wsize):
                            variables.append(self.convert_index(w_x + x, w_y + y))

        indexes = [[b, var] for b in range(bs) for var in variables]
        # bs * internal variables * variable for each internal variable (neighbours + self)
        windows = ((rows - 2) // (self.wsize - 2)) * ((cols - 2) // (self.wsize - 2))
        assert len(indexes) == (bs * self.wsize * self.wsize * windows)
        for index in indexes:
            sample, var_index = index
            assert var_index < (rows * cols), var_index
            assert var_index >= 0
        return indexes

    def _forward(self, **graph_nodes):
        """
        Returns the semantic loss related to the instance of this class, using the G_output_logit node.
        The sigmoid function is applied by the loss if the property use_sigmoid is set to True.

        :param graph_nodes: Dict of tf graph nodes, G_output_logit must be present.
        :return: A pair of dicts, where the first one is for giving nodes to the trainer, the second is for providing
        nodes that we want to be logged.
        The first dict maps:
        - "G_loss" to the semantic loss of this class
        - self.__class__.name to the semantic loss of this class
        - self.__class__.name + "_wmc" to the wmc defined by the vtree and sdd for this class
        The second dict maps:
        - "G_loss" to the semantic loss of this class
        - self.__class__.name to the semantic loss of this class
        - self.__class__.name + "_wmc" to the wmc defined by the vtree and sdd for this class
        - self.__class__.name + "_wmc_per_sample" to the wmc defined by the vtree and sdd for this class, over each
            sample.
        """
        self.logger.info("Semantic loss %s " % self.constraint_name)
        g_output_logit = graph_nodes["G_output_logit"]
        self.logger.info("Semantic loss found G_output_logit of shape %s" % g_output_logit.shape)

        # set values to probabilities
        toprobs = g_output_logit
        if self.use_sigmoid:
            toprobs = tf.nn.sigmoid(g_output_logit, name=self.constraint_name + "_to_probabilities")

        bs = self.experiment["BATCH_SIZE"]
        total_variables = reduce(lambda x, y: x * y, toprobs.shape[1:])
        toprobs = tf.reshape(toprobs, [bs, total_variables])
        self.logger.info("Semantic loss reshaping toprobs to %s" % toprobs.shape)
        shape = self.experiment["SHAPE"]
        rows = shape[0]
        cols = shape[1]

        # get corner variables
        corner_indexes = self.corner_indexes()
        corner_vars = tf.gather_nd(toprobs, corner_indexes)
        self.logger.info("Semantic loss gathered corner variables shape %s" % corner_vars.shape)
        # reshape as [bs * # corners, variables per corner]
        corner_vars = tf.reshape(corner_vars, [bs * 4, 4])
        self.logger.info("Semantic loss gathered corner variables reshaped to %s" % corner_vars.shape)

        # get edge variables
        edge_indexes = self.edge_indexes()
        edge_vars = tf.gather_nd(toprobs, edge_indexes)
        self.logger.info("Semantic loss gathered edge  variables shape %s" % edge_vars.shape)
        # reshape as [bs * edge variables, variables per edge variable]
        edge_vars = tf.reshape(edge_vars, [bs * ((rows - 2) * 2 + ((cols - 2) * 2)), 6])
        self.logger.info("Semantic loss gathered edge variables reshaped to %s" % edge_vars.shape)

        # get internal variables
        internal_indexes = self.window_indexes()
        internal_vars = tf.gather_nd(toprobs, internal_indexes)
        self.logger.info("Semantic loss gathered internal  variables shape %s" % internal_vars.shape)
        # reshape as [bs * internal variables, variables per internal variable]
        windows = ((rows - 2) // (self.wsize - 2)) * ((cols - 2) // (self.wsize - 2))
        internal_vars = tf.reshape(internal_vars, [bs * windows, self.wsize ** 2])
        self.logger.info("Semantic loss gathered internal variables reshaped to %s" % internal_vars.shape)

        """
        For each type of group of variables run it through the related
        arithmetic circuit.
        """
        self.logger.info("Importing psdd still_life_corners")
        corner_psdd = self.__class__._import_psdd("still_life_corners")
        corner_wmc = corner_psdd.generate_tf_ac_v2(corner_vars)
        self.logger.info("Semantic loss corner wmc shape %s" % corner_wmc.shape)
        # reshape to [bs, number of corners]
        corner_wmc = tf.reshape(corner_wmc, [bs, 4])
        self.logger.info("Semantic loss corner wmc reshaped to %s" % corner_wmc.shape)
        del corner_psdd

        self.logger.info("Importing psdd still_life_edges")
        edge_psdd = self.__class__._import_psdd("still_life_edges")
        edge_wmc = edge_psdd.generate_tf_ac_v2(edge_vars)
        self.logger.info("Semantic loss edge wmc shape %s" % edge_wmc.shape)
        # reshape to [bs, number of edge variables]
        edge_wmc = tf.reshape(edge_wmc, [bs, ((rows - 2) * 2 + (cols - 2) * 2)])
        self.logger.info("Semantic loss edge wmc reshaped to %s" % edge_wmc.shape)
        del edge_psdd

        self.logger.info("Importing psdd still_life_window_%s" % self.wsize)
        internal_psdd = self.__class__._import_psdd("still_life_window_%s" % self.wsize)
        internal_wmc = internal_psdd.generate_tf_ac_v2(internal_vars)
        self.logger.info("Semantic loss internal wmc shape %s" % internal_wmc.shape)
        # reshape to [bs, number of internal variables]
        internal_wmc = tf.reshape(internal_wmc, [bs, windows])
        self.logger.info("Semantic loss internal wmc reshaped to %s" % internal_wmc.shape)
        del internal_psdd

        """
        Stuff related to the same batch sample should be on the same dimension, i.e. for sample i
        we find wmcs related to it in corner_wmc[i,...], edge_wmc[i,...] and internal_wmc[i,...], concat
        everything on axis1 then reduce prod to apply an "AND" to these constraints.
        """
        wmc = tf.concat([corner_wmc, edge_wmc, internal_wmc], axis=1)

        # mask out a certain rate of constraints to be "ANDED" to avoid gradient vanishing
        mask = tf.random_uniform(wmc.shape)
        thres = self.experiment["SEMANTIC_LOSS_CONSTRAINTS_DROPOUT_RATE"]
        if self.experiment["SEMANTIC_LOSS_CONSTRAINTS_DROPOUT_DECREMENTAL"]:
            self.logger.info("Using decrement over time for constraints dropout")
            weight = 1.0 - (tf.cast(self._get_current_epoch(graph_nodes) + 1, tf.float32) / (
                    self.experiment["LEARNING_EPOCHS"] + 1))
            thres = thres * weight

        mask1 = tf.cast((mask >= thres), tf.float32) * wmc
        mask0 = tf.cast((mask < thres), tf.float32) * tf.ones(wmc.shape)
        wmc = mask1 + mask0

        self.logger.info("Semantic loss concatenating on axis=1 corner, edge, internal wmcs, "
                         "obtained shape %s" % wmc.shape)
        wmc_per_sample = tf.reduce_prod(wmc, axis=1)
        self.logger.info("Semantic loss reduced product wmc shape %s" % wmc_per_sample.shape)
        wmc = tf.reduce_mean(wmc_per_sample)
        self.logger.info("Semantic loss reduced mean wmc shape %s" % wmc.shape)
        semantic_loss_pre_timing = -tf.log(wmc + 1e-8)

        if self.experiment["SL_EQUALIZE"]:
            alpha = tf.abs(tf.stop_gradient(graph_nodes["G_adversarial_loss"] / semantic_loss_pre_timing))
            semantic_loss_pre_timing *= alpha

        semantic_loss = self._time_semantic_loss(graph_nodes, semantic_loss_pre_timing)




        # fill dict and return
        nodes = dict()
        nodes["G_loss"] = semantic_loss  # needed cos its expected by the trainer
        # add these nodes so that statistics can pick them up later if needed
        nodes["SemanticLoss_" + self.constraint_name] = semantic_loss_pre_timing
        nodes["SemanticLoss_" + self.constraint_name + "_wmc"] = wmc
        nodes["SemanticLoss_" + self.constraint_name + "_wmc_per_sample"] = wmc_per_sample
        nodes_to_log = {**nodes}
        nodes_to_log.pop("SemanticLoss_" + self.constraint_name + "_wmc_per_sample")

        return nodes, nodes_to_log, dict()


###############################################################################
################### RANDOM FORMULA GENERATION #################################
###############################################################################


class SemanticLoss_synthetic_formula(_SemanticLoss):

    def __init__(self, experiment, *args, **kwargs):
        super().__init__(experiment, *args, **kwargs)
        self.sample_n_vars = \
        self.experiment["SHAPE"][0]
        self.constraint_name = self.experiment["FORMULA_FILE"]


    def _import_psdd(self):
        """
        Given a constraint_name, assert the existence and look for the related .vtree and .sdd files, which
        are expected to be in in/semantic_loss_constraints/constraints_as_vtree and /constraints_as_sdd
        respectively, and to be named as constraint_name.vtree and .sdd respectively.
        The vtree and sdd are loaded and used to instantiate the psdd, which is then returned.
        :param constraint_name: Name of the constraints to use.
        """
        cwd = os.getcwd()
        assert cwd[-4:] == "/src", "Expected to be in the src directory of the repository, the cwd is instead %s" % cwd

        vtree = "in/semantic_loss_constraints/constraints_as_vtree/" + self.constraint_name + ".vtree"
        sdd = "in/semantic_loss_constraints/constraints_as_sdd/" + self.constraint_name + ".sdd"
        assert os.path.isfile(vtree), vtree + " is not a file."
        assert os.path.isfile(sdd), sdd + " is not a file."

        # load vtree and sdd files and construct the PSDD
        vtree = Vtree.read(vtree)
        manager = SddManager(vtree)
        alpha = io.sdd_read(sdd, manager)
        pmanager = PSddManager(vtree)
        psdd = pmanager.copy_and_normalize_sdd(alpha, vtree)

        return psdd


    def _forward(self, **graph_nodes):
        """
        Returns the semantic loss related to the instance of this class, using the G_output_logit node.
        The sigmoid function is applied by the loss if the property use_sigmoid is set to True.

        :param graph_nodes: Dict of tf graph nodes, G_output_logit must be present.
        :return: A pair of dicts, where the first one is for giving nodes to the trainer, the second is for providing
        nodes that we want to be logged.
        The first dict maps:
        - "G_loss" to the semantic loss of this class
        - self.__class__.name to the semantic loss of this class
        - self.__class__.name + "_wmc" to the wmc defined by the vtree and sdd for this class
        The second dict maps:
        - "G_loss" to the semantic loss of this class
        - self.__class__.name to the semantic loss of this class
        - self.__class__.name + "_wmc" to the wmc defined by the vtree and sdd for this class
        - self.__class__.name + "_wmc_per_sample" to the wmc defined by the vtree and sdd for this class, over each
            sample.
        """
        node_name = "synthetic_formula"
        self.logger.info("Semantic loss %s " % self.constraint_name)
        self.logger.info("Importing psdd %s " % self.constraint_name)
        psdd = self._import_psdd()
        g_output_logit = graph_nodes["G_output_logit"]
        self.logger.info(
            "Semantic loss found G_output_logit of shape %s" % g_output_logit.shape)

        # set values to probabilities
        toprobs = g_output_logit
        if self.use_sigmoid:
            toprobs = tf.nn.sigmoid(g_output_logit,
                                    name=self.constraint_name + "_to_probabilities")


        wmc = psdd.generate_tf_ac_v2(toprobs)
        self.logger.info("Semantic loss wmc of shape %s" % wmc.shape)
        wmc_per_sample = wmc
        self.logger.info(
            "Semantic loss wmc reduce product to shape %s" % wmc_per_sample.shape)
        wmc = tf.reduce_mean(wmc_per_sample)
        self.logger.info(
            "Semantic loss wmc reduced mean to shape %s" % wmc.shape)
        semantic_loss_pre_timing = -tf.log(wmc)
        semantic_loss = self._time_semantic_loss(graph_nodes,
                                                 semantic_loss_pre_timing)

        # fill dict and return
        nodes = dict()
        nodes[
            "G_loss"] = semantic_loss  # needed cos its expected by the trainer
        # add these nodes so that statistics can pick them up later if needed
        nodes[
            "SemanticLoss_" + node_name] = semantic_loss_pre_timing
        nodes["SemanticLoss_" + node_name + "_wmc"] = wmc
        nodes[
            "SemanticLoss_" + node_name + "_wmc_per_sample"] = wmc_per_sample
        nodes_to_log = {**nodes}
        nodes_to_log.pop(
            "SemanticLoss_" + node_name + "_wmc_per_sample")

        del psdd

        return nodes, nodes_to_log, dict()

####################################################################
################### LEVEL GENERATION LOSSES ########################
####################################################################

class _LevelSemanticLoss(_SemanticLoss):

    def __init__(self, experiment, *args, **kwargs):
        super().__init__(experiment, *args, **kwargs)

        self.use_sigmoid = True
        self.sample_height, self.sample_width, self.sample_channels = self.experiment["SHAPE"]

    def _add_SemanticLoss_statistic_nodes(self, nodes, original_loss, timed_loss):
        nodes["G_semantic_loss_original_" + self.constraint_name] = original_loss
        nodes["G_semantic_loss_timed_" + self.constraint_name] = timed_loss

    def _add_VNU_statistic_nodes(self, nodes, original_loss, wmc, wmc_per_sample):
        nodes["SemanticLoss_" + self.constraint_name] = original_loss
        nodes["SemanticLoss_" + self.constraint_name + "_wmc"] = wmc
        nodes["SemanticLoss_" + self.constraint_name + "_wmc_per_sample"] = wmc_per_sample

    '''
    def _post_processing(self, new_nodes, nodes_to_log, nodes_to_init):
        weight = tf.constant(self._weight, dtype=tf.float32)
        zero = tf.constant(0.0, dtype=tf.float32)

        # G_loss must always be present, no problem in non-checking its presence in new_nodes
        self.logger.info(f"Adding 0 condition on node G_loss in constraint {self.constraint_name}, dict new_nodes")
        new_nodes["G_loss"] = tf.cond(tf.equal(weight, zero), true_fn=lambda: zero, false_fn=lambda: new_nodes["G_loss"])

        # optional nodes
        names = [
            "G_semantic_loss_original_" + self.constraint_name,
            "G_semantic_loss_timed_" + self.constraint_name,
            "SemanticLoss_" + self.constraint_name,
            "SemanticLoss_" + self.constraint_name + "_wmc",
            "SemanticLoss_" + self.constraint_name + "_wmc_per_sample"
        ]
        for name in names:
            if name in new_nodes:
                self.logger.info(f"Adding 0 condition on node {name} in constraint {self.constraint_name}, dict new_nodes")
                new_nodes[name] = tf.cond(tf.equal(weight, zero), true_fn=lambda: zero, false_fn=lambda: new_nodes[name])
            if name in nodes_to_log:
                self.logger.info(f"Adding 0 condition on node {name} in constraint {self.constraint_name}, dict nodes_to_log")
                nodes_to_log[name] = tf.cond(tf.equal(weight, zero), true_fn=lambda: zero, false_fn=lambda: nodes_to_log[name])
        
        # _post_processing allows no return value, dicts has to be edited in place
    '''


class SemanticLoss_onehot(_LevelSemanticLoss):

    def _forward(self, **graph_nodes):
        """
        Returns the semantic loss related to the instance of this class, using the G_output_logit node.
        The sigmoid function is applied by the loss if the property use_sigmoid is set to True.

        :param graph_nodes: Dict of tf graph nodes, G_output_logit must be present.
        :return: A pair of dicts, where the first one is for giving nodes to the trainer, the second is for providing
        nodes that we want to be logged.
        The first dict maps:
        - "G_loss" to the semantic loss of this class
        - self.__class__.name to the semantic loss of this class
        - self.__class__.name + "_wmc" to the wmc defined by the vtree and sdd for this class
        The second dict maps:
        - "G_loss" to the semantic loss of this class
        - self.__class__.name to the semantic loss of this class
        - self.__class__.name + "_wmc" to the wmc defined by the vtree and sdd for this class
        - self.__class__.name + "_wmc_per_sample" to the wmc defined by the vtree and sdd for this class, over each
            sample.
        """
        self.logger.info("Semantic loss %s " % self.constraint_name)
        self.logger.info("Importing psdd %s " % self.constraint_name)
        psdd = self.__class__._import_psdd(self.constraint_name)
        G_output_logit = graph_nodes["G_output_logit"]
        self.logger.info("Semantic loss found G_output_logit of shape %s" % G_output_logit.shape)

        # G_sample are already probabilities
        toprobs = G_output_logit
        self.logger.info("Semantic loss {} needs normalization: {}".format(self.__class__.__name__, self.use_sigmoid))
        if self.use_sigmoid:
            toprobs = tf.nn.softmax(G_output_logit, axis=-1, name=self.constraint_name + "_to_probabilities")

        # reshaping to [batch_size * height * width, sample_channels]
        aggregate = tf.reshape(toprobs, shape=[-1, self.sample_channels])
        self.logger.info("Semantic loss reshaping variables to shape %s" % aggregate.shape)
        # generating logic tf tree
        wmc = psdd.generate_tf_ac_v2(aggregate)
        self.logger.info("Semantic loss wmc of shape %s" % wmc.shape)
        # reshaping to [batch_size, height * width]
        wmc = tf.reshape(wmc, [-1, (self.sample_height * self.sample_width)])
        self.logger.info("Semantic loss wmc reshaped to shape %s" % wmc.shape)
        wmc_per_sample = tf.reduce_prod(wmc, axis=1)
        self.logger.info("Semantic loss wmc reduce product to shape %s" % wmc_per_sample.shape)
        wmc = tf.reduce_mean(wmc_per_sample)
        self.logger.info("Semantic loss wmc reduced mean to shape %s" % wmc.shape)

        semantic_loss_pre_timing = -tf.log(tf.maximum(wmc, tf.constant(EPSILON)))
        semantic_loss = self._time_semantic_loss(graph_nodes, semantic_loss_pre_timing)

        # building results
        nodes = dict()
        # most important key cos it's expected by the trainer
        nodes["G_loss"] = semantic_loss
        # adding stuff for statistics
        self._add_SemanticLoss_statistic_nodes(nodes, semantic_loss_pre_timing, semantic_loss)
        # add these nodes so that VNU statistics can pick them up later if needed
        self._add_VNU_statistic_nodes(nodes, semantic_loss_pre_timing, wmc, wmc_per_sample)

        # add these nodes so that they will be logged in terminal
        nodes_to_log = dict()
        self._add_SemanticLoss_statistic_nodes(nodes_to_log, semantic_loss_pre_timing, semantic_loss)

        del psdd
        return nodes, nodes_to_log, dict()


class SemanticLoss_pipes(_LevelSemanticLoss):

    def __init__(self, experiment, *args, **kwargs):
        super().__init__(experiment, *args, **kwargs)

        # retrieve pipe tiles indexes
        self.target_channels = [self.experiment["TILES_MAP"][x]['index'] for x in ['<', '>', '[', ']']]
        self.other_channels = [self.experiment["TILES_MAP"][x]['index'] for x in ['X', 'S', '-', '?', 'Q', 'E', 'o', 'B', 'b']]

    def _forward(self, **graph_nodes):
        """
        Returns the semantic loss related to the instance of this class, using the G_output_logit node.
        The sigmoid function is applied by the loss if the property use_sigmoid is set to True.

        :param graph_nodes: Dict of tf graph nodes, G_output_logit must be present.
        :return: A pair of dicts, where the first one is for giving nodes to the trainer, the second is for providing
        nodes that we want to be logged.
        The first dict maps:
        - "G_loss" to the semantic loss of this class
        - self.__class__.name to the semantic loss of this class
        - self.__class__.name + "_wmc" to the wmc defined by the vtree and sdd for this class
        The second dict maps:
        - "G_loss" to the semantic loss of this class
        - self.__class__.name to the semantic loss of this class
        - self.__class__.name + "_wmc" to the wmc defined by the vtree and sdd for this class
        - self.__class__.name + "_wmc_per_sample" to the wmc defined by the vtree and sdd for this class, over each
            sample.
        """
        self.logger.info("Semantic loss %s " % self.constraint_name)
        self.logger.info("Importing psdd %s " % self.constraint_name)
        psdd = self.__class__._import_psdd(self.constraint_name)
        G_output_logit = graph_nodes["G_output_logit"]
        self.logger.info("Semantic loss found G_output_logit of shape %s" % G_output_logit.shape)

        # G_sample are already probabilities
        toprobs = G_output_logit
        self.logger.info("Semantic loss {} needs normalization: {}".format(self.__class__.__name__, self.use_sigmoid))
        if self.use_sigmoid:
            toprobs = tf.nn.softmax(G_output_logit, axis=-1, name=self.constraint_name + "_to_probabilities")

        # extract the 4 channels used by this constraint: <, >, [, ]
        pipes_channels = tf.gather(toprobs, self.target_channels, axis=-1)
        other_channels_together = tf.reduce_sum(tf.gather(toprobs, self.other_channels, axis=-1), axis=-1, keepdims=True)

        # last channel becomes <, >, [, ], others
        toprobs = tf.concat([pipes_channels, other_channels_together], axis=-1)

        # pipes are like the following, without a fixed height
        #
        #  <>
        #  []    <>
        #  []    []     <>
        #  []    []     []
        ####################

        # splitting levels in blocks of [2, 2, 5]
        squares_array = []
        # toprobs has shape (batch_size, height, width, channels)
        for i in range(self.sample_height - 1):
            for j in range(self.sample_width - 1):
                squares_array.append(
                    toprobs[:, i:i+2, j:j+2, :]
                )
        # stacking on axis 1 to keep squares of the same array compact
        aggregate = tf.stack(squares_array, axis=1)
        # reshaping to merge first two dimensions
        # aggregate = tf.reshape(aggregate, shape=(-1, 2, 2, 5))
        # aggregate has shape: [batch_size * (constraints_per_sample), 2, 2, 5]

        # reshaping to [batch_size * (constraints_per_sample), 20]
        aggregate = tf.reshape(aggregate, shape=(-1, 20))
        self.logger.info("Semantic loss reshaping variables to shape %s" % aggregate.shape)

        """
        Once we have the aggregate [batch_size * (height-1) * width, 16] tensor pass it thought the SL,
        reduce_prod on constraints related to different rows/cols of the same data sample (as if we were doing
        an AND), then reduce mean over the samples.
        """

        # generate tf tree that encodes semantic loss
        wmc = psdd.generate_tf_ac_v2(aggregate)

        self.logger.info("Semantic loss wmc of shape %s" % wmc.shape)
        # now reshaping to [batch_size, constraints_per_sample]
        wmc = tf.reshape(wmc, [-1, (self.sample_height - 1) * (self.sample_width - 1)])
        self.logger.info("Semantic loss wmc reshaped to shape %s" % wmc.shape)
        wmc_per_sample = tf.reduce_prod(wmc, axis=1)
        self.logger.info("Semantic loss wmc reduce product to shape %s" % wmc_per_sample.shape)
        wmc = tf.reduce_mean(wmc_per_sample)
        self.logger.info("Semantic loss wmc reduced mean to shape %s" % wmc.shape)
        semantic_loss_pre_timing = -tf.log(tf.maximum(wmc, tf.constant(EPSILON)))
        semantic_loss = self._time_semantic_loss(graph_nodes, semantic_loss_pre_timing)

        # building results
        nodes = dict()
    
        # most important key cos it's expected by the trainer
        nodes["G_loss"] = semantic_loss
        # adding stuff for statistics
        self._add_SemanticLoss_statistic_nodes(nodes, semantic_loss_pre_timing, semantic_loss)
        # add these nodes so that VNU statistics can pick them up later if needed
        self._add_VNU_statistic_nodes(nodes, semantic_loss_pre_timing, wmc, wmc_per_sample)

        # add these nodes so that they will be logged in terminal
        nodes_to_log = dict()
        self._add_SemanticLoss_statistic_nodes(nodes_to_log, semantic_loss_pre_timing, semantic_loss)

        del psdd
        return nodes, nodes_to_log, dict()

    
class SemanticLoss_all_mario(_LevelSemanticLoss):

    def __init__(self, experiment, *args, **kwargs):
        super().__init__(experiment, *args, **kwargs)

        # retrieve pipe tiles indexes
        self.pipes_tiles = [self.experiment["TILES_MAP"][x]['index'] for x in ['<', '>', '[', ']']]
        self.solid_tiles = [self.experiment["TILES_MAP"][x]['index'] for x in ['X', 'S', '?', 'Q']]
        self.air_tiles = [self.experiment["TILES_MAP"][x]['index'] for x in ['-', 'o']]
        self.cannons_tiles = [self.experiment["TILES_MAP"][x]['index'] for x in ['B', 'b']]
        self.enemy_tile = [self.experiment["TILES_MAP"][x]['index'] for x in ['E']]

    def _forward(self, **graph_nodes):
        """
        Returns the semantic loss related to the instance of this class, using the G_output_logit node.
        The sigmoid function is applied by the loss if the property use_sigmoid is set to True.

        :param graph_nodes: Dict of tf graph nodes, G_output_logit must be present.
        :return: A pair of dicts, where the first one is for giving nodes to the trainer, the second is for providing
        nodes that we want to be logged.
        The first dict maps:
        - "G_loss" to the semantic loss of this class
        - self.__class__.name to the semantic loss of this class
        - self.__class__.name + "_wmc" to the wmc defined by the vtree and sdd for this class
        The second dict maps:
        - "G_loss" to the semantic loss of this class
        - self.__class__.name to the semantic loss of this class
        - self.__class__.name + "_wmc" to the wmc defined by the vtree and sdd for this class
        - self.__class__.name + "_wmc_per_sample" to the wmc defined by the vtree and sdd for this class, over each
            sample.
        """
        self.logger.info("Semantic loss %s " % self.constraint_name)
        self.logger.info("Importing psdd %s " % self.constraint_name)
        psdd = self.__class__._import_psdd(self.constraint_name)
        G_output_logit = graph_nodes["G_output_logit"]
        self.logger.info("Semantic loss found G_output_logit of shape %s" % G_output_logit.shape)

        # G_sample are already probabilities
        toprobs = G_output_logit
        self.logger.info("Semantic loss {} needs normalization: {}".format(self.__class__.__name__, self.use_sigmoid))
        if self.use_sigmoid:
            toprobs = tf.nn.softmax(G_output_logit, axis=-1, name=self.constraint_name + "_to_probabilities")

        # order and group channels
        pipes_channels = tf.gather(toprobs, self.pipes_tiles, axis=-1)
        monster_channel = tf.gather(toprobs, self.enemy_tile, axis=-1)
        cannons_channels = tf.gather(toprobs, self.cannons_tiles, axis=-1)
        air_tiles = tf.reduce_sum(tf.gather(toprobs, self.air_tiles, axis=-1), axis=-1, keepdims=True)
        solid_tiles = tf.reduce_sum(tf.gather(toprobs, self.solid_tiles, axis=-1), axis=-1, keepdims=True)

        self.logger.info("Semantic loss found pipes_channels of shape {}".format(pipes_channels.shape))
        self.logger.info("Semantic loss found monster_channel of shape {}".format(monster_channel.shape))
        self.logger.info("Semantic logic loss found cannons_channels of shape {}".format(cannons_channels.shape))
        self.logger.info("Semantic logic loss found air_tiles of shape {}".format(air_tiles.shape))
        self.logger.info("Semantic logic loss found solid_tiles of shape {}".format(solid_tiles.shape))

        # channels order becomes "<" | ">" | "[" | "]" | "E" | "B" | "b" | "XSQE" | "-o"
        toprobs = tf.concat([pipes_channels, monster_channel, cannons_channels, solid_tiles, air_tiles], axis=-1)

        # splitting levels in blocks of [2, 2, 9]
        squares_array = []
        # toprobs has shape (batch_size, height, width, channels)
        for i in range(self.sample_height - 1):
            for j in range(self.sample_width - 1):
                squares_array.append(
                    toprobs[:, i:i+2, j:j+2, :]
                )
        # stacking on axis 1 to keep squares of the same array compact
        aggregate = tf.stack(squares_array, axis=1)
        # reshaping to merge first two dimensions
        # aggregate = tf.reshape(aggregate, shape=(-1, 2, 2, 9))
        # aggregate has shape: [batch_size * (constraints_per_sample), 2, 2, 9]

        # reshaping to [batch_size * (constraints_per_sample), 36]
        aggregate = tf.reshape(aggregate, shape=(-1, 36))
        self.logger.info("Semantic loss reshaping variables to shape %s" % aggregate.shape)

        """
        Once we have the aggregate [batch_size * (height-1) * (width-1), 36] tensor pass it thought the SL,
        reduce_prod on constraints related to different rows/cols of the same data sample (as if we were doing
        an AND), then reduce mean over the samples.
        """

        # generate tf tree that encodes semantic loss
        wmc = psdd.generate_tf_ac_v2(aggregate)

        self.logger.info("Semantic loss wmc of shape %s" % wmc.shape)
        # now reshaping to [batch_size, constraints_per_sample]
        wmc = tf.reshape(wmc, [-1, (self.sample_height - 1) * (self.sample_width - 1)])
        self.logger.info("Semantic loss wmc reshaped to shape %s" % wmc.shape)
        wmc_per_sample = tf.reduce_prod(wmc, axis=1)
        self.logger.info("Semantic loss wmc reduce product to shape %s" % wmc_per_sample.shape)
        wmc = tf.reduce_mean(wmc_per_sample)
        self.logger.info("Semantic loss wmc reduced mean to shape %s" % wmc.shape)
        semantic_loss_pre_timing = -tf.log(tf.maximum(wmc, tf.constant(EPSILON)))
        semantic_loss = self._time_semantic_loss(graph_nodes, semantic_loss_pre_timing)

        # building results
        nodes = dict()

        # semantic_loss = tf.Print(semantic_loss, [semantic_loss], message="passing per the fucking dio semantic loss nonostante dovrebbe essere tagliata fora")

        # most important key cos it's expected by the trainer
        nodes["G_loss"] = semantic_loss
        # adding stuff for statistics
        self._add_SemanticLoss_statistic_nodes(nodes, semantic_loss_pre_timing, semantic_loss)
        # add these nodes so that VNU statistics can pick them up later if needed
        self._add_VNU_statistic_nodes(nodes, semantic_loss_pre_timing, wmc, wmc_per_sample)

        # add these nodes so that they will be logged in terminal
        nodes_to_log = dict()
        self._add_SemanticLoss_statistic_nodes(nodes_to_log, semantic_loss_pre_timing, semantic_loss)

        del psdd
        return nodes, nodes_to_log, dict()



class SemanticLoss_reachability(_LevelSemanticLoss):

    def __init__(self, experiment, *args, **kwargs):
        super().__init__(experiment, *args, **kwargs)

        # get the channel indexes of air-like and solid-like tiles
        self.air = [self.experiment["TILES_MAP"][x]['index'] for x in ['-', 'E', 'o']] # air, monsters and coins can be traversed
        self.solid = [self.experiment["TILES_MAP"][x]['index'] for x in ['X', 'S', '?', 'Q', '[', ']', '<', '>', 'B', 'b']]

        # check that experiments contains all the necessary options to define the CNN network
        assert self.experiment["REACHABILITY_ARCHITECTURE"] is not None, "A reachability architecture is required to use SemanticLoss_reachability"
        assert self.experiment["REACHABILITY_PRETRAINED_MODEL"] is not None, "A reachability model should be provided"

    def _forward(self, **graph_nodes):
        """
        Returns the semantic loss related to the instance of this class, using the G_output_logit node.

        :param graph_nodes: Dict of tf graph nodes, G_output_logit must be present.
        :return: A triple of dicts, where the first one is for giving nodes to the trainer, the second is for providing
        nodes that we want to be logged and the third one is to provide architectures that need an initialization
        The first dict maps:
        - "G_loss" to the semantic loss of this class
        - self.__class__.name to the semantic loss of this class
        - self.__class__.name + "_wmc" to the wmc defined by the vtree and sdd for this class
        The second dict maps:
        - "G_loss" to the semantic loss of this class
        - self.__class__.name to the semantic loss of this class
        - self.__class__.name + "_wmc" to the wmc defined by the vtree and sdd for this class
        - self.__class__.name + "_wmc_per_sample" to the wmc defined by the vtree and sdd for this class, over each
            sample.
        The third dict contains the reachability CNN that has to be restored from model and initialized before the training.
        """
        # dict of results
        nodes = dict()
        # dict of nodes that has to be initialized
        nodes_to_init = dict()

        # first of all, load the cnn network and add restoring of pre-trained weights to dict of nodes that has to be init
        if "reachability_cnn" in graph_nodes:
            # if the networks was used before, use the same instance
            self.logger.info("Reachability network has already ben initialized, using existing instance")
            reachability_cnn = graph_nodes["reachability_cnn"]
        else:
            # if the reachability networks has not yet being initialized
            self.logger.info("Reachability network not found, creating a new one")
            reachability_cnn = self.experiment["REACHABILITY_ARCHITECTURE"](self.experiment, trainable=False)
            reachability_cnn.model_path = self.experiment["REACHABILITY_PRETRAINED_MODEL"]
            # allow future instances of this class to use this reachability cnn
            nodes["reachability_cnn"] = reachability_cnn
            # need initialization of pre-trained weights
            nodes_to_init["reachability_cnn"] = reachability_cnn

        self.logger.info("Semantic loss %s " % self.constraint_name)
        self.logger.info("Importing psdd %s " % self.constraint_name)
        psdd = self.__class__._import_psdd(self.constraint_name)
        G_output_logit = graph_nodes["G_output_logit"]
        self.logger.info("Semantic loss found G_output_logit of shape %s" % G_output_logit.shape)

        # G_sample are already probabilities
        toprobs = G_output_logit
        self.logger.info("Semantic loss {} needs normalization: {}".format(self.__class__.__name__, self.use_sigmoid))
        if self.use_sigmoid:
            toprobs = tf.nn.softmax(G_output_logit, axis=-1, name=self.constraint_name + "_to_probabilities")

        # summing up all the blocks that are solid-like and all the passable ones
        G_probab_solid = tf.reduce_sum(tf.gather(toprobs, self.solid, axis=-1), axis=-1, keep_dims=True)
        # G_probab_solid = tf.expand_dims(G_probab_solid, axis=-1)
        # allow statistics to access this tensor
        graph_nodes["G_probab_solid"] = G_probab_solid
        # getting reachability map
        results, _, _ = reachability_cnn(**graph_nodes)
        aggregate = results["reachability_map"]
        aggregate = tf.nn.softmax(aggregate, axis=-1)

        # aggregate: [batch_size, height, width, 1 (reachable)]
        aggregate = tf.gather(aggregate[:, -2:, :, 1], [0, self.sample_width-1], axis=2)
        # aggregate: [batch_size, 2, 2, 1]

        # reshaping to [batch_size, 2 * 2 * 1]
        aggregate = tf.reshape(aggregate, shape=[-1, 2 * 2 * 1])
        self.logger.info("Semantic loss reshaping variables to shape %s" % aggregate.shape)
        # generating logic tf tree
        wmc = psdd.generate_tf_ac_v2(aggregate)
        self.logger.info("Semantic loss wmc of shape %s" % wmc.shape)
        # reshaping to [batch_size, 1]
        wmc = tf.reshape(wmc, [-1, 1])
        self.logger.info("Semantic loss wmc reshaped to shape %s" % wmc.shape)
        wmc_per_sample = tf.reduce_prod(wmc, axis=1)
        self.logger.info("Semantic loss wmc reduce product to shape %s" % wmc_per_sample.shape)
        wmc = tf.reduce_mean(wmc_per_sample)
        self.logger.info("Semantic loss wmc reduced mean to shape %s" % wmc.shape)

        semantic_loss_pre_timing = -tf.log(tf.maximum(wmc, tf.constant(EPSILON)))
        semantic_loss = self._time_semantic_loss(graph_nodes, semantic_loss_pre_timing)

        # most important key cos it's expected by the trainer
        nodes["G_loss"] = semantic_loss
        # adding stuff for statistics
        self._add_SemanticLoss_statistic_nodes(nodes, semantic_loss_pre_timing, semantic_loss)
        # add these nodes so that VNU statistics can pick them up later if needed
        self._add_VNU_statistic_nodes(nodes, semantic_loss_pre_timing, wmc, wmc_per_sample)
        # adding pre-vtree
        # nodes["reachability_aggregate"] = aggregate

        # add these nodes so that they will be logged in terminal
        nodes_to_log = dict()
        self._add_SemanticLoss_statistic_nodes(nodes_to_log, semantic_loss_pre_timing, semantic_loss)

        del psdd
        return nodes, nodes_to_log, nodes_to_init


class SemanticLoss_reachability_astar(_LevelSemanticLoss):

    def __init__(self, experiment, *args, **kwargs):
        super().__init__(experiment, *args, **kwargs)

        self.air = [self.experiment["TILES_MAP"][x]['index'] for x in ['-', 'E', 'o']]
        self.solid = [self.experiment["TILES_MAP"][x]['index'] for x in ['X', 'S', '?', 'Q', '[', ']', '<', '>', 'B', 'b']]

        # check that experiments contains all the necessary options to define the CNN network
        assert self.experiment["REACHABILITY_ARCHITECTURE"] is not None, "A reachability architecture is required to use SemanticLoss_reachability"
        assert self.experiment["REACHABILITY_PRETRAINED_MODEL"] is not None, "A reachability model should be provided"

    def _forward(self, **graph_nodes):
        """
        Returns the semantic loss related to the instance of this class, using the G_output_logit node.

        :param graph_nodes: Dict of tf graph nodes, G_output_logit must be present.
        :return: A triple of dicts, where the first one is for giving nodes to the trainer, the second is for providing
        nodes that we want to be logged and the third one is to provide architectures that need an initialization
        The first dict maps:
        - "G_loss" to the semantic loss of this class
        - self.__class__.name to the semantic loss of this class
        - self.__class__.name + "_wmc" to the wmc defined by the vtree and sdd for this class
        The second dict maps:
        - "G_loss" to the semantic loss of this class
        - self.__class__.name to the semantic loss of this class
        - self.__class__.name + "_wmc" to the wmc defined by the vtree and sdd for this class
        - self.__class__.name + "_wmc_per_sample" to the wmc defined by the vtree and sdd for this class, over each
            sample.
        The third dict contains the reachability CNN that has to be restored from model and initialized before the training.
        """
        # dict of results
        nodes = dict()
        # dict of nodes that has to be initialized
        nodes_to_init = dict()

        # first of all, load the cnn network and add restoring of pre-trained weights to dict of nodes that has to be init
        if "reachability_cnn" in graph_nodes:
            # if the networks was used before, use the same instance
            self.logger.info("Reachability network has already ben initialized, using existing instance")
            reachability_cnn = graph_nodes["reachability_cnn"]
        else:
            # if the reachability networks has not yet being initialized
            self.logger.info("Reachability network not found, creating a new one")
            reachability_cnn = self.experiment["REACHABILITY_ARCHITECTURE"](self.experiment, trainable=False)
            reachability_cnn.model_path = self.experiment["REACHABILITY_PRETRAINED_MODEL"]
            # allow future instances of this class to use this reachability cnn
            nodes["reachability_cnn"] = reachability_cnn
            # need initialization of pre-trained weights
            nodes_to_init["reachability_cnn"] = reachability_cnn

        self.logger.info("Semantic loss %s " % self.constraint_name)
        self.logger.info("Importing psdd %s " % self.constraint_name)
        psdd = self.__class__._import_psdd(self.constraint_name)
        G_output_logit = graph_nodes["G_output_logit"]
        self.logger.info("Semantic loss found G_output_logit of shape %s" % G_output_logit.shape)

        # G_sample are already probabilities
        toprobs = G_output_logit
        self.logger.info("Semantic loss {} needs normalization: {}".format(self.__class__.__name__, self.use_sigmoid))
        if self.use_sigmoid:
            toprobs = tf.nn.softmax(G_output_logit, axis=-1, name=self.constraint_name + "_to_probabilities")

        # summing up all the blocks that are solid-like and all the passable ones
        G_probab_solid = tf.reduce_sum(tf.gather(toprobs, self.solid, axis=-1), axis=-1)
        G_probab_solid = tf.expand_dims(G_probab_solid, axis=-1)
        # use by statistics
        graph_nodes["G_probab_solid"] = G_probab_solid
        # getting reachability map
        aggregate = reachability_cnn(**graph_nodes)
        aggregate = tf.nn.softmax(aggregate, axis=-1)

        # aggregate: [batch_size, height, width, 1 (reachable)]
        aggregate = aggregate[:, :, -1, 1]
        # aggregate: [batch_size, 14]

        self.logger.info("Semantic loss reshaping variables to shape %s" % aggregate.shape)
        # generating logic tf tree
        wmc = psdd.generate_tf_ac_v2(aggregate)
        self.logger.info("Semantic loss wmc of shape %s" % wmc.shape)
        # reshaping to [batch_size, 1]
        wmc = tf.reshape(wmc, [-1, 1])
        self.logger.info("Semantic loss wmc reshaped to shape %s" % wmc.shape)
        wmc_per_sample = tf.reduce_prod(wmc, axis=1)
        self.logger.info("Semantic loss wmc reduce product to shape %s" % wmc_per_sample.shape)
        wmc = tf.reduce_mean(wmc_per_sample)
        self.logger.info("Semantic loss wmc reduced mean to shape %s" % wmc.shape)

        semantic_loss_pre_timing = -tf.log(tf.maximum(wmc, tf.constant(EPSILON)))
        semantic_loss = self._time_semantic_loss(graph_nodes, semantic_loss_pre_timing)

        # most important key cos it's expected by the trainer
        nodes["G_loss"] = semantic_loss
        # adding stuff for statistics
        self._add_SemanticLoss_statistic_nodes(nodes, semantic_loss_pre_timing, semantic_loss)
        # add these nodes so that VNU statistics can pick them up later if needed
        self._add_VNU_statistic_nodes(nodes, semantic_loss_pre_timing, wmc, wmc_per_sample)
        # adding pre-vtree
        # nodes["reachability_aggregate"] = aggregate

        # add these nodes so that they will be logged in terminal
        nodes_to_log = dict()
        self._add_SemanticLoss_statistic_nodes(nodes_to_log, semantic_loss_pre_timing, semantic_loss)

        del psdd
        return nodes, nodes_to_log, nodes_to_init


class SemanticLoss_stairs(_LevelSemanticLoss):

    def __init__(self, experiment, *args, **kwargs):
        super().__init__(experiment, *args, **kwargs)

        self.air = [self.experiment["TILES_MAP"][x]['index'] for x in ['-', 'E', 'o']]
        self.solid = [self.experiment["TILES_MAP"][x]['index'] for x in ['X', 'S', '?', 'Q', '[', ']', '<', '>', 'B', 'b']]

    def _forward(self, **graph_nodes):
        """
        Returns the semantic loss related to the instance of this class, using the G_output_logit node.

        :param graph_nodes: Dict of tf graph nodes, G_output_logit must be present.
        :return: A triple of dicts, where the first one is for giving nodes to the trainer, the second is for providing
        nodes that we want to be logged and the third one is to provide architectures that need an initialization
        The first dict maps:
        - "G_loss" to the semantic loss of this class
        - self.__class__.name to the semantic loss of this class
        - self.__class__.name + "_wmc" to the wmc defined by the vtree and sdd for this class
        The second dict maps:
        - "G_loss" to the semantic loss of this class
        - self.__class__.name to the semantic loss of this class
        - self.__class__.name + "_wmc" to the wmc defined by the vtree and sdd for this class
        - self.__class__.name + "_wmc_per_sample" to the wmc defined by the vtree and sdd for this class, over each
            sample.
        The third dict contains the reachability CNN that has to be restored from model and initialized before the training.
        """
        self.logger.info("Semantic loss %s " % self.constraint_name)
        self.logger.info("Importing psdd %s " % self.constraint_name)
        psdd = self.__class__._import_psdd(self.constraint_name)
        G_output_logit = graph_nodes["G_output_logit"]
        self.logger.info("Semantic loss found G_output_logit of shape %s" % G_output_logit.shape)

        # G_sample are already probabilities
        toprobs = G_output_logit
        if self.use_sigmoid:
            toprobs = tf.nn.softmax(G_output_logit, axis=-1, name=self.constraint_name + "_to_probabilities")

        # summing up all the blocks that are solid-like
        G_probab_solid = tf.reduce_sum(tf.gather(toprobs, self.solid, axis=-1), axis=-1)

        # summing up all the blocks that are air-like
        G_probab_air = tf.reduce_sum(tf.gather(toprobs, self.air, axis=-1), axis=-1)

        # stacking air and solid probabilities on the last axis
        aggregate = tf.stack([G_probab_air, G_probab_solid], axis=-1)
        # G_probab_air_solid has shape [batch_size, height, width, 2-[p_air, p_solid]]

        # TODO: choose the position where you want to put the staircase or stack together different windows
        # to requires multiple stairs. Windows shoud have size [batch_size, 4, 4, 2]
        # at the moment using a window in the bottom left part of the level.
        aggregate = aggregate[:, -5:-1, 1:5, :]
        # aggregate shape: [batch_size, 4, 4, 2]

        # reshaping to [batch_size, 32]
        aggregate = tf.reshape(aggregate, shape=[-1, 4*4*2])
        self.logger.info("Semantic loss reshaping variables to shape %s" % aggregate.shape)
        # generating logic tf tree
        wmc = psdd.generate_tf_ac_v2(aggregate)
        self.logger.info("Semantic loss wmc of shape %s" % wmc.shape)
        # reshaping to [batch_size, 1] -> 1 constraint for each level
        wmc = tf.reshape(wmc, [-1, 1])
        self.logger.info("Semantic loss wmc reshaped to shape %s" % wmc.shape)
        wmc_per_sample = tf.reduce_prod(wmc, axis=1)
        self.logger.info("Semantic loss wmc reduce product to shape %s" % wmc_per_sample.shape)
        wmc = tf.reduce_mean(wmc_per_sample)
        self.logger.info("Semantic loss wmc reduced mean to shape %s" % wmc.shape)

        semantic_loss_pre_timing = -tf.log(tf.maximum(wmc, tf.constant(EPSILON)))
        semantic_loss = self._time_semantic_loss(graph_nodes, semantic_loss_pre_timing)

        # fill dict and return
        nodes = dict()

        # add these nodes so that statistics can pick them up later if needed
        nodes["SemanticLoss_" + self.constraint_name] = semantic_loss_pre_timing
        nodes["SemanticLoss_" + self.constraint_name + "_wmc"] = wmc
        nodes["SemanticLoss_" + self.constraint_name + "_wmc_per_sample"] = wmc_per_sample

        nodes["G_loss"] = semantic_loss
        nodes["G_semantic_loss_original"] = semantic_loss_pre_timing
        nodes["G_semantic_loss_timed"] = semantic_loss

        nodes_to_log = dict()
        #nodes_to_log.pop("SemanticLoss_" + self.constraint_name + "_wmc_per_sample")
        nodes_to_log["G_semantic_loss_wmc"] = wmc
        nodes_to_log["G_semantic_loss_original"] = semantic_loss_pre_timing
        nodes_to_log["G_semantic_loss_timed"] = semantic_loss

        del psdd
        return nodes, nodes_to_log, dict()




class SemanticLoss_reachability_row3of7(_LevelSemanticLoss):

    def __init__(self, experiment, *args, **kwargs):
        super().__init__(experiment, *args, **kwargs)

        self.air = [self.experiment["TILES_MAP"][x]['index'] for x in ['-', 'E', 'o']]
        self.solid = [self.experiment["TILES_MAP"][x]['index'] for x in ['X', 'S', '?', 'Q', '[', ']', '<', '>', 'B', 'b']]

        # check that experiments contains all the necessary options to define the CNN network
        assert self.experiment["REACHABILITY_ARCHITECTURE"] is not None, "A reachability architecture is required to use SemanticLoss_reachability"
        assert self.experiment["REACHABILITY_PRETRAINED_MODEL"] is not None, "A reachability model should be provided"

    def _forward(self, **graph_nodes):
        """
        Returns the semantic loss related to the instance of this class, using the G_output_logit node.

        :param graph_nodes: Dict of tf graph nodes, G_output_logit must be present.
        :return: A triple of dicts, where the first one is for giving nodes to the trainer, the second is for providing
        nodes that we want to be logged and the third one is to provide architectures that need an initialization
        The first dict maps:
        - "G_loss" to the semantic loss of this class
        - self.__class__.name to the semantic loss of this class
        - self.__class__.name + "_wmc" to the wmc defined by the vtree and sdd for this class
        The second dict maps:
        - "G_loss" to the semantic loss of this class
        - self.__class__.name to the semantic loss of this class
        - self.__class__.name + "_wmc" to the wmc defined by the vtree and sdd for this class
        - self.__class__.name + "_wmc_per_sample" to the wmc defined by the vtree and sdd for this class, over each
            sample.
        The third dict contains the reachability CNN that has to be restored from model and initialized before the training.
        """
        # dict of results
        nodes = dict()
        # dict of nodes that has to be initialized
        nodes_to_init = dict()

        # first of all, load the cnn network and add restoring of pre-trained weights to dict of nodes that has to be init
        if "reachability_cnn" in graph_nodes:
            # if the networks was used before, use the same instance
            self.logger.info("Reachability network has already ben initialized, using existing instance")
            reachability_cnn = graph_nodes["reachability_cnn"]
        else:
            # if the reachability networks has not yet being initialized
            self.logger.info("Reachability network not found, creating a new one")
            reachability_cnn = self.experiment["REACHABILITY_ARCHITECTURE"](self.experiment, trainable=False)
            reachability_cnn.model_path = self.experiment["REACHABILITY_PRETRAINED_MODEL"]
            # allow future instances of this class to use this reachability cnn
            nodes["reachability_cnn"] = reachability_cnn
            # need initialization of pre-trained weights
            nodes_to_init["reachability_cnn"] = reachability_cnn

        self.logger.info("Semantic loss %s " % self.constraint_name)
        self.logger.info("Importing psdd %s " % self.constraint_name)
        psdd = self.__class__._import_psdd(self.constraint_name)
        G_output_logit = graph_nodes["G_output_logit"]
        self.logger.info("Semantic loss found G_output_logit of shape %s" % G_output_logit.shape)

        # G_sample are already probabilities
        toprobs = G_output_logit
        self.logger.info("Semantic loss {} needs normalization: {}".format(self.__class__.__name__, self.use_sigmoid))
        if self.use_sigmoid:
            toprobs = tf.nn.softmax(G_output_logit, axis=-1, name=self.constraint_name + "_to_probabilities")

        # summing up all the blocks that are solid-like and all the passable ones
        G_probab_solid = tf.reduce_sum(tf.gather(toprobs, self.solid, axis=-1), axis=-1)
        G_probab_solid = tf.expand_dims(G_probab_solid, axis=-1)

        graph_nodes["G_probab_solid"] = G_probab_solid

        aggregate = reachability_cnn(**graph_nodes)
        aggregate = tf.nn.softmax(aggregate, axis=-1)

        # take only the reachable channel, the softmax layer SHOULD adjust the unreachable channel accordingly...
        aggregate = aggregate[:, :, :, 1]
        # aggregate: [batch_size, height, width]
        # extracting the whole 5th row from the bottom and divide it in four pieces
        aggregate = aggregate[:, 9, :]
        # aggregate: [batch_size, width]
        aggregate = tf.stack(tf.split(aggregate, 4, axis=-1), axis=1)

        # aggregate: [batch_size, 4, 7]
        # for each sample there are four pieces of the 5th rows with lenght 7
        # reshaping to [batch_size * 4, 7]
        aggregate = tf.reshape(aggregate, shape=[-1, 7])
        self.logger.info("Semantic loss reshaping variables to shape %s" % aggregate.shape)
        # generating logic tf tree
        wmc = psdd.generate_tf_ac_v2(aggregate)
        self.logger.info("Semantic loss wmc of shape %s" % wmc.shape)
        # reshaping to [batch_size, 4] # 4 constraints per sample
        wmc = tf.reshape(wmc, [-1, 4])
        self.logger.info("Semantic loss wmc reshaped to shape %s" % wmc.shape)
        wmc_per_sample = tf.reduce_prod(wmc, axis=1)
        self.logger.info("Semantic loss wmc reduce product to shape %s" % wmc_per_sample.shape)
        wmc = tf.reduce_mean(wmc_per_sample)
        self.logger.info("Semantic loss wmc reduced mean to shape %s" % wmc.shape)

        semantic_loss_pre_timing = -tf.log(tf.maximum(wmc, tf.constant(EPSILON)))
        semantic_loss = self._time_semantic_loss(graph_nodes, semantic_loss_pre_timing)

        # most important key cos it's expected by the trainer
        nodes["G_loss"] = semantic_loss
        # adding stuff for statistics
        self._add_SemanticLoss_statistic_nodes(nodes, semantic_loss_pre_timing, semantic_loss)
        # add these nodes so that VNU statistics can pick them up later if needed
        self._add_VNU_statistic_nodes(nodes, semantic_loss_pre_timing, wmc, wmc_per_sample)

        # add these nodes so that they will be logged in terminal
        nodes_to_log = dict()
        self._add_SemanticLoss_statistic_nodes(nodes_to_log, semantic_loss_pre_timing, semantic_loss)

        del psdd
        return nodes, nodes_to_log, nodes_to_init


class SemanticLoss_monsters(_LevelSemanticLoss):

    def __init__(self, experiment, *args, **kwargs):
        super().__init__(experiment, *args, **kwargs)
        self.monster_channels = [self.experiment["TILES_MAP"][x]['index'] for x in ['E']]
        self.solid_channels = [self.experiment["TILES_MAP"][x]['index'] for x in ['X', 'S', '?', 'Q', '<', '>', 'B']]
        self.other_channels = [self.experiment["TILES_MAP"][x]['index'] for x in ['-', '[', ']', 'o', 'b']]

    def _forward(self, **graph_nodes):
        """
        Returns the semantic loss related to the instance of this class, using the G_output_logit node.
        The sigmoid function is applied by the loss if the property use_sigmoid is set to True.

        :param graph_nodes: Dict of tf graph nodes, G_output_logit must be present.
        :return: A pair of dicts, where the first one is for giving nodes to the trainer, the second is for providing
        nodes that we want to be logged.
        The first dict maps:
        - "G_loss" to the semantic loss of this class
        - self.__class__.name to the semantic loss of this class
        - self.__class__.name + "_wmc" to the wmc defined by the vtree and sdd for this class
        The second dict maps:
        - "G_loss" to the semantic loss of this class
        - self.__class__.name to the semantic loss of this class
        - self.__class__.name + "_wmc" to the wmc defined by the vtree and sdd for this class
        - self.__class__.name + "_wmc_per_sample" to the wmc defined by the vtree and sdd for this class, over each
            sample.
        """
        self.logger.info("Semantic loss %s " % self.constraint_name)
        self.logger.info("Importing psdd %s " % self.constraint_name)
        psdd = self.__class__._import_psdd(self.constraint_name)
        G_output_logit = graph_nodes["G_output_logit"]
        self.logger.info("Semantic loss found G_output_logit of shape %s" % G_output_logit.shape)

        toprobs = G_output_logit
        if self.use_sigmoid:
            toprobs = tf.nn.softmax(G_output_logit, axis=-1, name=self.constraint_name + "_to_probabilities")

        # extracting cannons channels
        monsters_channels = tf.gather(toprobs, self.monster_channels, axis=-1)
        solid_channels = tf.reduce_sum(tf.gather(toprobs, self.solid_channels, axis=-1), axis=-1, keepdims=True)
        other_channels = tf.reduce_sum(tf.gather(toprobs, self.other_channels, axis=-1), axis=-1, keepdims=True)

        # channels: E | solid | others
        toprobs = tf.concat([monsters_channels, solid_channels, other_channels], axis=-1)

        # splitting levels in blocks of [2, 1, 3]
        rectangles_array = []
        # aggregate has shape (batch_size or less, height - 1, width, channels)
        for i in range(self.sample_height - 1):
            for j in range(self.sample_width):
                rectangles_array.append(
                    toprobs[:, i:i+2, j:j+1, :]
                )

        aggregate = tf.stack(rectangles_array, axis=1)
        # aggregate has shape [batch_size, height * width, 2, 1, 3]

        aggregate = tf.reshape(aggregate, shape=(-1, 6))
        # aggregate: [batch_size * height * width, 6]

        self.logger.info("Semantic loss reshaping variables to shape %s" % aggregate.shape)

        """
        Once we have the aggregate [batch_size * (height-1) * width, 4] tensor pass it thought the circuit,
        reduce_prod on constraints related to different rows/cols of the same data sample (as if we were doing
        an AND), then reduce mean over the samples.
        """
        wmc = psdd.generate_tf_ac_v2(aggregate)
        self.logger.info("Semantic loss wmc of shape %s" % wmc.shape)
        wmc = tf.reshape(wmc, [-1, (self.sample_height - 1) * self.sample_width])
        self.logger.info("Semantic loss wmc reshaped to shape %s" % wmc.shape)
        wmc_per_sample = tf.reduce_prod(wmc, axis=1)
        self.logger.info("Semantic loss wmc reduce product to shape %s" % wmc_per_sample.shape)
        wmc = tf.reduce_mean(wmc_per_sample)
        self.logger.info("Semantic loss wmc reduced mean to shape %s" % wmc.shape)
        semantic_loss_pre_timing = -tf.log(tf.maximum(wmc, tf.constant(EPSILON)))
        semantic_loss = self._time_semantic_loss(graph_nodes, semantic_loss_pre_timing)

        # fill dict and return
        nodes = dict()

        # needed cos its expected by the trainer
        nodes["G_loss"] = semantic_loss
        # adding stuff for statistics
        self._add_SemanticLoss_statistic_nodes(nodes, semantic_loss_pre_timing, semantic_loss)
        # add these nodes so that VNU statistics can pick them up later if needed
        self._add_VNU_statistic_nodes(nodes, semantic_loss_pre_timing, wmc, wmc_per_sample)

        # add these nodes so that they will be logged in terminal
        nodes_to_log = dict()
        self._add_SemanticLoss_statistic_nodes(nodes_to_log, semantic_loss_pre_timing, semantic_loss)
    
        del psdd
        return nodes, nodes_to_log, dict()


class SemanticLoss_cannons(_LevelSemanticLoss):

    def __init__(self, experiment, *args, **kwargs):
        super().__init__(experiment, *args, **kwargs)
        self.cannons_channels = [self.experiment["TILES_MAP"][x]['index'] for x in ['b', 'B']]
        self.solid_channels = [self.experiment["TILES_MAP"][x]['index'] for x in ['X', 'S', '?', 'Q']]
        self.other_channels = [self.experiment["TILES_MAP"][x]['index'] for x in ['-', 'o', 'b', '<', '>', '[', ']']]

    def _forward(self, **graph_nodes):
        """
        Returns the semantic loss related to the instance of this class, using the G_output_logit node.
        The sigmoid function is applied by the loss if the property use_sigmoid is set to True.

        :param graph_nodes: Dict of tf graph nodes, G_output_logit must be present.
        :return: A pair of dicts, where the first one is for giving nodes to the trainer, the second is for providing
        nodes that we want to be logged.
        The first dict maps:
        - "G_loss" to the semantic loss of this class
        - self.__class__.name to the semantic loss of this class
        - self.__class__.name + "_wmc" to the wmc defined by the vtree and sdd for this class
        The second dict maps:
        - "G_loss" to the semantic loss of this class
        - self.__class__.name to the semantic loss of this class
        - self.__class__.name + "_wmc" to the wmc defined by the vtree and sdd for this class
        - self.__class__.name + "_wmc_per_sample" to the wmc defined by the vtree and sdd for this class, over each
            sample.
        """
        self.logger.info(f"Semantic loss {self.constraint_name}")
        self.logger.info(f"Importing psdd {self.constraint_name}")
        psdd = self.__class__._import_psdd(self.constraint_name)
        G_output_logit = graph_nodes["G_output_logit"]
        self.logger.info(f"Semantic loss found G_output_logit of shape {G_output_logit.shape}")

        toprobs = G_output_logit
        if self.use_sigmoid:
            toprobs = tf.nn.softmax(G_output_logit, axis=-1, name=self.constraint_name + "_to_probabilities")

        # extracting cannons channels
        cannons_channels = tf.gather(toprobs, self.cannons_channels, axis=-1)
        solid_channels = tf.reduce_sum(tf.gather(toprobs, self.solid_channels, axis=-1), axis=-1, keepdims=True)
        other_channels = tf.reduce_sum(tf.gather(toprobs, self.other_channels, axis=-1), axis=-1, keepdims=True)

        # channels: b | B | solid | others
        toprobs = tf.concat([cannons_channels, solid_channels, other_channels], axis=-1)

        # splitting levels in blocks of [2, 1, 4]
        rectangles_array = []
        # aggregate has shape (batch_size or less, height - 1, width, channels)
        for i in range(self.sample_height - 1):
            for j in range(self.sample_width):
                rectangles_array.append(
                    toprobs[:, i:i+2, j:j+1, :]
                )

        # stacking on axis 1 to keep squares of the same array near
        aggregate = tf.stack(rectangles_array, axis=1)
        # aggregate has shape [batch_size, (height - 1) * width, 2, 1, 4]

        aggregate = tf.reshape(aggregate, shape=(-1, 8))
        # aggregate: [batch_size * (height - 1) * width, 8]

        self.logger.info("Semantic loss reshaping variables to shape {}".format(aggregate.shape))

        """
        Once we have the aggregate [batch_size * (height-1) * width, 8] tensor pass it thought the circuit,
        reduce_prod on constraints related to different rows/cols of the same data sample (as if we were doing
        an AND), then reduce mean over the samples.
        """

        wmc = psdd.generate_tf_ac_v2(aggregate)
        self.logger.info("Semantic loss wmc of shape %s" % wmc.shape)
        wmc = tf.reshape(wmc, [-1, (self.sample_height - 1) * self.sample_width])
        self.logger.info("Semantic loss wmc reshaped to shape %s" % wmc.shape)
        wmc_per_sample = tf.reduce_prod(wmc, axis=1)
        self.logger.info("Semantic loss wmc reduce product to shape %s" % wmc_per_sample.shape)
        wmc = tf.reduce_mean(wmc_per_sample)
        self.logger.info("Semantic loss wmc reduced mean to shape %s" % wmc.shape)
        semantic_loss_pre_timing = -tf.log(tf.maximum(wmc, tf.constant(EPSILON)))
        semantic_loss = self._time_semantic_loss(graph_nodes, semantic_loss_pre_timing)

        # fill dict and return
        nodes = dict()
        # needed cos its expected by the trainer
        nodes["G_loss"] = semantic_loss
        
        # adding stuff for statistics
        self._add_SemanticLoss_statistic_nodes(nodes, semantic_loss_pre_timing, semantic_loss)
        # add these nodes so that VNU statistics can pick them up later if needed
        self._add_VNU_statistic_nodes(nodes, semantic_loss_pre_timing, wmc, wmc_per_sample)

        # add these nodes so that they will be logged in terminal
        nodes_to_log = dict()
        self._add_SemanticLoss_statistic_nodes(nodes_to_log, semantic_loss_pre_timing, semantic_loss)

        del psdd
        return nodes, nodes_to_log, dict()




"""
Code down here is used to create all constraint classes in an automatic way, adding vtree and sdd files
in their relative directory is enough (you don't have to do anything more code wise).
What happens is that for every constraint for which both vtree and sdd file are present, a new class with the name
SemanticLoss_<name of the constraint is created, which inherits from semantic loss.
"""


def _create_semantic_loss_class(name):
    """
    Given a class name, create a class type with that name which inherits from SemanticLoss
    :param name: Name of the classe to create.
    """
    attributes_dict = {"__init__": lambda self, experiment: _SemanticLoss.__init__(self, experiment)}
    _tmpclass = type(name, (_SemanticLoss,), attributes_dict)
    globals()[_tmpclass.__name__] = _tmpclass
    del _tmpclass


"""
For each constraint create a class.
This workaround allows to simply add vtree and sdd files to add a new constraints, without
having to touch up the losses module every time.
"""

_constraints = _get_constraints_names()
for c in _constraints:
    name = "SemanticLoss_%s" % c
    if name not in globals():
        _create_semantic_loss_class(name)
    del name
del _constraints
