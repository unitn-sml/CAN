import os
import tensorflow as tf
from fuzzy.parser import Parser
from base_layers import Loss
from fuzzy.lyrics import fuzzy
from utils import utils_common
import numpy as np
from fuzzy.util import _get_constraints_names


class _FuzzyLogicLoss(Loss):
    """
    Given a boolean formula expressed as 
    """

    def _pre_processing(self, **graph_nodes):
        assert "G_output_logit" in graph_nodes, "Expected to find G_output_logit in graph nodes, which is the raw " \
                                                "output of the Generator (This loss will take care of applying the " \
                                                "sigmoid function)"

    def __init__(self, experiment, *args, **kwargs):
        super().__init__(experiment, *args, **kwargs)

        self.logger = self.experiment["LOGGER"]

        # if the input is probabilities already don't use sigmoid on it, otherwise use sigmoid
        self._use_sigmoid = not self.experiment["FUZZY_LOGIC_LOSS_INPUT_IS_PROBABILITIES"]

        self.constraint_name = self.__class__.__name__
        assert self.constraint_name[:15] == "FuzzyLogicLoss_", "Expected FuzzyLogicLoss_ as prefix of the name of a" \
            " class extending FuzzyLogicLoss, the name is instead {}".format(self.constraint_name)
        self.constraint_name = "_".join(self.__class__.__name__.split("_")[1:])

        self.fuzzy_class = self.experiment["FUZZY_LOGIC_CIRCUIT"]
        
        fuzzy_logic_loss_from = experiment["FUZZY_LOGIC_LOSS_FROM_EPOCH"]
        self.fuzzy_logic_from = fuzzy_logic_loss_from if fuzzy_logic_loss_from is not None else 0
        self.fuzzy_logic_incremental = bool(experiment["FUZZY_LOGIC_LOSS_INCREMENTAL"])

    def _get_current_epoch(self, graph_nodes):
        """
        Get the current epoch as a tf node, automatically updates
        if G_global_step is used by any optimizer.

        :param graph_nodes:
        :return:
        """
        return graph_nodes["current_epoch"]

    def _incremental_timing(self, graph_nodes, fuzzy_loss):
        """
        If experiment["FUZZY_LOGIC_LOSS_INCREMENTAL"] is true then the fuzzylogic loss is adjusted with a weight
        equal to (current epoch + 1)/(total epochs + 1), otherwise the fuzzylogic loss is returned as it is.

        :param graph_nodes:
        :param fuzzylogic_loss:
        :return:
        """
        if self.fuzzy_logic_incremental:
            self.logger.info("Using incremental fuzzy loss")
            weight = tf.cast(
                self._get_current_epoch(graph_nodes) + 1 - self.fuzzy_logic_from,
                tf.float32
            ) / tf.cast(self.experiment["LEARNING_EPOCHS"] + 1 - self.fuzzy_logic_from, tf.float32)
            return fuzzy_loss * weight
        else:
            self.logger.info("Not using incremental fuzzy loss")
            return fuzzy_loss

    def _time_fuzzy_loss(self, graph_nodes, fuzzy_loss):
        """
        Get the fuzzy loss as a tf node which value is zero if the current epoch is lower
        than the experiment["FUZZY_LOGIC_LOSS_FROM_EPOCH"] parameter, otherwise the fuzzy loss is returned.

        :param graph_nodes:
        :param fuzzy_loss:
        :return:
        """
        self.logger.info("Using fuzzy logic loss starting from epochs {}".format(self.fuzzy_logic_from))
        cond = self._get_current_epoch(graph_nodes) >= self.fuzzy_logic_from
        truefn = lambda: fuzzy_loss
        falsefn = lambda: tf.constant(0.)
        epoch_timed = tf.cond(cond, true_fn=truefn, false_fn=falsefn)
        incremental_timed = self._incremental_timing(graph_nodes, epoch_timed)
        return incremental_timed

    @staticmethod
    def _import_sympy(experiment, constraint_name, fuzzy_class):
        """
        Given a constraint_name, assert the existence and look for the respective .sympy file which
        is expected to be in in/semantic_loss_constraints/constraints_as_sympy_tree
        and to be named as constraint_name.sympy and constraint_name.shape
        """
        cwd = os.getcwd()
        assert cwd[-4:] == "/src", "Expected to be in the src directory of the repository, the cwd is instead %s" % cwd

        constraint_name = constraint_name.replace("FuzzyLogicLoss_", "")
        if not experiment["USE_DNF"]:
            sympy_file = os.path.join("in", "semantic_loss_constraints",
                "constraints_as_sympy_tree", "{}.fuzzy".format(constraint_name))
        else:
            sympy_file = os.path.join("in", "semantic_loss_constraints",
                                      "constraints_as_sympy_tree",
                                      "{}.fuzzy".format(constraint_name))
        assert os.path.isfile(sympy_file), "{} is not a file.".format(sympy_file)

        # load vtree and sdd files and construct the PSDD
        parser = Parser(experiment, sympy_file, fuzzy_class)

        return parser

    def _forward(self, **graph_nodes):
        """
        Returns the fuzzy logic loss related to the instance of this class, using the G_output_logit node.
        The sigmoid function is applied by the loss if the property use_sigmoid is set to True.

        :param graph_nodes: Dict of tf graph nodes, G_output_logit must be present.
        :return: A pair of dicts, where the first one is for giving nodes to the trainer, the second is for providing
        nodes that we want to be logged.
        The first dict maps:
        - "G_loss" to the fuzzy logic loss of this class
        - self.__class__.name to the fuzzy logic loss of this class
        - self.__class__.name + "_wmc" to the wmc defined by the files for this class
        The second dict maps:
        - "G_loss" to the fuzzy logic loss of this class
        - self.__class__.name to the fuzzy logic loss of this class
        - self.__class__.name + "_wmc" to the wmc defined by the files for this class
        - self.__class__.name + "_wmc_per_sample" to the wmc defined by the files for this class, over each
            sample.
        """
        self.logger.info("Fuzzy logic loss {}".format(self.constraint_name))
        self.logger.info("Importing parser {}".format(self.constraint_name))
        parser = self.__class__._import_sympy(self.experiment, self.constraint_name, self.fuzzy_class)
        g_output_logit = graph_nodes["G_output_logit"]
        self.logger.info("Fuzzy logic loss found G_output_logit of shape {}".format(g_output_logit.shape))

        # set values to probabilities
        toprobs = g_output_logit
        if self.use_sigmoid:
            if "SHAPE" in self.experiment and self.experiment["SHAPE"][-1] > 2:
                toprobs = tf.nn.softmax(g_output_logit, name=self.constraint_name + "_to_probabilities")
            else:
                toprobs = tf.nn.sigmoid(g_output_logit, name=self.constraint_name + "_to_probabilities")

        # need to reshape as a 1d vector of variables for each sample, needed by parser for the tf AC
        nodes = dict()
        wmc_per_sample = parser.generate_tf_tree(toprobs)
        self.logger.info("Fuzzy logic loss wmc of shape {}".format(wmc_per_sample.shape))
        wmc = tf.reduce_mean(wmc_per_sample)
        self.logger.info("Fuzzy logic loss reduced wmc of shape {}".format(wmc.shape))
        fuzzy_logic_loss_pre_timing = 1 - wmc
        fuzzy_logic_loss = self._time_fuzzy_loss(graph_nodes, fuzzy_logic_loss_pre_timing)

        nodes["G_loss"] = fuzzy_logic_loss  # needed cos its expected by the trainer
        nodes[self.constraint_name] = fuzzy_logic_loss_pre_timing
        nodes[self.constraint_name + "_wmc"] = wmc
        nodes[self.constraint_name + "_wmc_per_sample"] = wmc_per_sample
        nodes_to_log = {**nodes}
        nodes_to_log.pop(self.constraint_name + "_wmc_per_sample")

        del parser
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



###############################################################################
######################## TOY DATASET #####################################
###############################################################################


class _SyntheticFuzzyLogicLoss(_FuzzyLogicLoss):
    
    def __init__(self, experiment, *args, **kwargs):
        super().__init__(experiment, *args, **kwargs)

        self.use_sigmoid = not self.experiment["FUZZY_LOGIC_LOSS_INPUT_IS_PROBABILITIES"]
        self.sample_height, self.sample_width, self.sample_channels = \
        self.experiment["SHAPE"]
        self.use_dnf = self.experiment["USE_DNF"]

    def _add_FuzzyLogicLoss_statistic_nodes(self, nodes, original_loss,
                                            timed_loss):
        nodes["G_fuzzy_logic_loss_original_" + self.constraint_name] = original_loss
        nodes["G_fuzzy_logic_loss_timed_" + self.constraint_name] = timed_loss

    def _add_VNU_statistic_nodes(self, nodes, original_loss, wmc,
                                 wmc_per_sample):
        nodes["FuzzyLogicLoss_" + self.constraint_name] = original_loss
        nodes["FuzzyLogicLoss_" + self.constraint_name + "_wmc"] = wmc
        nodes["FuzzyLogicLoss_" + self.constraint_name + "_wmc_per_sample"] = wmc_per_sample


class FuzzyLogicLoss_pc_custom_rows(_SyntheticFuzzyLogicLoss):

    def _forward(self, **graph_nodes):
        """
        Returns the fuzzy logic loss related to the instance of this class, using the G_output_logit node.
        The sigmoid function is applied by the loss if the property use_sigmoid is set to True.

        :param graph_nodes: Dict of tf graph nodes, G_output_logit must be present.
        :return: A pair of dicts, where the first one is for giving nodes to the trainer, the second is for providing
        nodes that we want to be logged.
        The first dict maps:
        - "G_loss" to the fuzzy logic loss of this class
        - self.__class__.name to the fuzzy logic loss of this class
        - self.__class__.name + "_wmc" to the wmc defined by the files for this class
        The second dict maps:
        - "G_loss" to the fuzzy logic loss of this class
        - self.__class__.name to the fuzzy logic loss of this class
        - self.__class__.name + "_wmc" to the wmc defined by the files for this class
        - self.__class__.name + "_wmc_per_sample" to the wmc defined by the files for this class, over each
            sample.
        """
        self.logger.info("Fuzzy logic loss {}".format(self.constraint_name))
        self.logger.info("Importing parser {}".format(self.constraint_name))
        parser = self.__class__._import_sympy(self.experiment, self.constraint_name, self.fuzzy_class)
        g_output_logit = graph_nodes["G_output_logit"]
        self.logger.info("Fuzzy logic loss found G_output_logit of shape {}".format(g_output_logit.shape))

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
        self.logger.info("Fuzzy logic loss reshaping variables to shape %s" % aggregate.shape)

        """
        Once we have the aggregate [batch_size * 18, 10] tensor pass it thought the circuit,
        reduce_prod on constraints related to different rows/cols of the same data sample (as if we were doing
        an AND), then reduce mean over the samples.
        """
        wmc = parser.generate_tf_tree(aggregate, self.use_dnf)
        self.logger.info("Fuzzy logic loss wmc of shape {}".format(wmc.shape))
        wmc = tf.reshape(wmc, [-1, 18 * 2])
        self.logger.info(
            "Fuzzy logic wmc reshaped to shape {}".format(wmc.shape))
        wmc_per_sample = tf.reduce_prod(wmc, axis=1)
        self.logger.info(
            "Fuzzy logic loss wmc reduce product to shape {}".format(
                wmc_per_sample.shape))
        wmc = tf.reduce_mean(wmc_per_sample)
        self.logger.info(
            "Fuzzy logic loss wmc reduced mean to shape {}".format(wmc.shape))
        fuzzy_logic_loss_pre_timing = 1 - wmc
        fuzzy_logic_loss = self._time_fuzzy_loss(graph_nodes,
                                                 fuzzy_logic_loss_pre_timing)

        # fill dict and return
        nodes = dict()

        # most important key cos it's expected by the trainer
        nodes["G_loss"] = fuzzy_logic_loss
        # adding stuff for statistics
        self._add_FuzzyLogicLoss_statistic_nodes(nodes,
                                                 fuzzy_logic_loss_pre_timing,
                                                 fuzzy_logic_loss)
        # add these nodes so that VNU statistics can pick them up later if needed
        self._add_VNU_statistic_nodes(nodes, fuzzy_logic_loss_pre_timing, wmc,
                                      wmc_per_sample)

        # add these nodes so that they will be logged in terminal
        nodes_to_log = dict()
        self._add_FuzzyLogicLoss_statistic_nodes(nodes_to_log,
                                                 fuzzy_logic_loss_pre_timing,
                                                 fuzzy_logic_loss)

        del parser
        return nodes, nodes_to_log, dict()



###############################################################################
################### RANDOM FORMULA GENERATION #################################
###############################################################################


class _SyntheticFormulaFuzzyLogicLoss(_FuzzyLogicLoss):
    def __init__(self, experiment, *args, **kwargs):
        super().__init__(experiment, *args, **kwargs)
        self.use_sigmoid = not self.experiment["FUZZY_LOGIC_LOSS_INPUT_IS_PROBABILITIES"]
        self.sample_n_vars = \
        self.experiment["SHAPE"][0]
        self.use_dnf = self.experiment["USE_DNF"]
        self.constraint_name = self.experiment["FORMULA_FILE"]

    def _add_FuzzyLogicLoss_statistic_nodes(self, nodes, original_loss,
                                            timed_loss):
        node_name = "synthetic_formula"
        nodes["G_fuzzy_logic_loss_original_" + node_name] = original_loss
        nodes["G_fuzzy_logic_loss_timed_" + node_name] = timed_loss

    def _add_VNU_statistic_nodes(self, nodes, original_loss, wmc,
                                 wmc_per_sample):
        node_name = "synthetic_formula"
        nodes["FuzzyLogicLoss_" + node_name] = original_loss
        nodes["FuzzyLogicLoss_" + node_name + "_wmc"] = wmc
        nodes["FuzzyLogicLoss_" + node_name + "_wmc_per_sample"] = wmc_per_sample


    def _import_sympy(self, experiment, constraint_name, fuzzy_class):
        """
        Given a constraint_name, assert the existence and look for the respective .sympy file which
        is expected to be in in/semantic_loss_constraints/constraints_as_sympy_tree
        and to be named as constraint_name.sympy and constraint_name.shape
        """
        cwd = os.getcwd()
        assert cwd[
               -4:] == "/src", "Expected to be in the src directory of the repository, the cwd is instead %s" % cwd

        constraint_name = constraint_name.replace("FuzzyLogicLoss_", "")
        if not experiment["USE_DNF"]:
            sympy_file = os.path.join("in", "semantic_loss_constraints",
                                      "constraints_as_sympy_tree","formulas_cnf",
                                      "{}.fuzzy".format(constraint_name))
        else:
            sympy_file = os.path.join("in", "semantic_loss_constraints",
                                      "constraints_as_sympy_tree","formulas_dnf",
                                      "{}_dnf.fuzzy".format(constraint_name))
        assert os.path.isfile(sympy_file), "{} is not a file.".format(
            sympy_file)

        # load vtree and sdd files and construct the PSDD
        parser = Parser(experiment, sympy_file, fuzzy_class)

        return parser


class FuzzyLogicLoss_synthetic_formula(_SyntheticFormulaFuzzyLogicLoss):

    def _forward(self, **graph_nodes):
        """
        Returns the fuzzy logic loss related to the instance of this class, using the G_output_logit node.
        The sigmoid function is applied by the loss if the property use_sigmoid is set to True.

        :param graph_nodes: Dict of tf graph nodes, G_output_logit must be present.
        :return: A pair of dicts, where the first one is for giving nodes to the trainer, the second is for providing
        nodes that we want to be logged.
        The first dict maps:
        - "G_loss" to the fuzzy logic loss of this class
        - self.__class__.name to the fuzzy logic loss of this class
        - self.__class__.name + "_wmc" to the wmc defined by the files for this class
        The second dict maps:
        - "G_loss" to the fuzzy logic loss of this class
        - self.__class__.name to the fuzzy logic loss of this class
        - self.__class__.name + "_wmc" to the wmc defined by the files for this class
        - self.__class__.name + "_wmc_per_sample" to the wmc defined by the files for this class, over each
            sample.
        """
        self.logger.info("Fuzzy logic loss {}".format(self.constraint_name))
        self.logger.info("Importing parser {}".format(self.constraint_name))
        parser = self._import_sympy(self.experiment, self.constraint_name, self.fuzzy_class)
        g_output_logit = graph_nodes["G_output_logit"]
        self.logger.info("Fuzzy logic loss found G_output_logit of shape {}".format(g_output_logit.shape))

        # set values to probabilities
        toprobs = g_output_logit
        if self.use_sigmoid:
            toprobs = tf.nn.sigmoid(g_output_logit, name=self.constraint_name + "_to_probabilities")

        """
        Get n_vars and squeeze 1 dimensions [batch_size,1,n_vars,1] -> [batch_size, n_vars].
        """
        squeezed_tensor = tf.squeeze(toprobs)
        aggregate = tf.reshape(squeezed_tensor, (-1, self.sample_n_vars))
        self.logger.info("Fuzzy logic loss reshaping variables to shape %s" % aggregate.shape)

        """
        Once we have the aggregate [batch_size, n_vars] tensor pass it thought the circuit,
        reduce_prod on constraints related to different rows/cols of the same data sample (as if we were doing
        an AND), then reduce mean over the samples.
        """
        fuzzy_per_sample = parser.generate_tf_tree(aggregate, self.use_dnf)
        self.logger.info("Fuzzy logic result of shape {}".format(fuzzy_per_sample.shape))
        fuzzy_final = tf.reduce_mean(fuzzy_per_sample)
        self.logger.info(
            "Fuzzy logic loss circuit reduced mean to shape {}".format(fuzzy_final.shape))
        fuzzy_logic_loss_pre_timing = 1 - fuzzy_final
        fuzzy_logic_loss = self._time_fuzzy_loss(graph_nodes,
                                                 fuzzy_logic_loss_pre_timing)

        # fill dict and return
        nodes = dict()

        # most important key cos it's expected by the trainer
        nodes["G_loss"] = fuzzy_logic_loss
        # adding stuff for statistics
        self._add_FuzzyLogicLoss_statistic_nodes(nodes,
                                                 fuzzy_logic_loss_pre_timing,
                                                 fuzzy_logic_loss)
        # add these nodes so that VNU statistics can pick them up later if needed
        self._add_VNU_statistic_nodes(nodes, fuzzy_logic_loss_pre_timing, fuzzy_final,
                                      fuzzy_per_sample)

        # add these nodes so that they will be logged in terminal
        nodes_to_log = dict()
        self._add_FuzzyLogicLoss_statistic_nodes(nodes_to_log,
                                                 fuzzy_logic_loss_pre_timing,
                                                 fuzzy_logic_loss)

        del parser
        return nodes, nodes_to_log, dict()


###############################################################################
######################## LEVEL GENERATION #####################################
###############################################################################

class _LevelFuzzyLogicLoss(_FuzzyLogicLoss):

    def __init__(self, experiment, *args, **kwargs):
        super().__init__(experiment, *args, **kwargs)

        self.use_sigmoid = True
        self.sample_height, self.sample_width, self.sample_channels = self.experiment["SHAPE"]
    
    def _add_FuzzyLogicLoss_statistic_nodes(self, nodes, original_loss, timed_loss):
        nodes["G_fuzzy_logic_loss_original_" + self.constraint_name] = original_loss
        nodes["G_fuzzy_logic_loss_timed_" + self.constraint_name] = timed_loss

    def _add_VNU_statistic_nodes(self, nodes, original_loss, wmc, wmc_per_sample):
        nodes["FuzzyLogicLoss_" + self.constraint_name] = original_loss
        nodes["FuzzyLogicLoss_" + self.constraint_name + "_wmc"] = wmc
        nodes["FuzzyLogicLoss_" + self.constraint_name + "_wmc_per_sample"] = wmc_per_sample

    '''
    def _post_processing(self, new_nodes, nodes_to_log, nodes_to_init):
        weight = tf.constant(self._weight, dtype=tf.float32)
        zero = tf.constant(0.0, dtype=tf.float32)

        # G_loss must always be present, no problem in non-checking its presence in new_nodes
        self.logger.info(f"Adding 0 condition on node G_loss in constraint {self.constraint_name}, dict new_nodes")
        new_nodes["G_loss"] = tf.cond(tf.equal(weight, zero), true_fn=lambda: zero, false_fn=lambda: new_nodes["G_loss"])

        # optional nodes
        names = [
            "G_fuzzy_logic_loss_original_" + self.constraint_name,
            "G_fuzzy_logic_loss_timed_" + self.constraint_name,
            "FuzzyLogicLoss_" + self.constraint_name,
            "FuzzyLogicLoss_" + self.constraint_name + "_wmc",
            "FuzzyLogicLoss_" + self.constraint_name + "_wmc_per_sample"
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


class FuzzyLogicLoss_monsters(_LevelFuzzyLogicLoss):

    def __init__(self, experiment, *args, **kwargs):
        super().__init__(experiment, *args, **kwargs)
        self.monster_channels = [self.experiment["TILES_MAP"][x]['index'] for x in ['E']]
        self.solid_channels = [self.experiment["TILES_MAP"][x]['index'] for x in ['X', 'S', '?', 'Q', '<', '>', 'B']]
        self.other_channels = [self.experiment["TILES_MAP"][x]['index'] for x in ['-', '[', ']', 'o', 'b']]

    def _forward(self, **graph_nodes):
        """
        Returns the fuzzy logic loss related to the instance of this class, using the G_output_logit node.
        The sigmoid function is applied by the loss if the property use_sigmoid is set to True.

        :param graph_nodes: Dict of tf graph nodes, G_output_logit must be present.
        :return: A pair of dicts, where the first one is for giving nodes to the trainer, the second is for providing
        nodes that we want to be logged.
        The first dict maps:
        - "G_loss" to the fuzzy logic loss of this class
        - self.__class__.name to the fuzzy logic loss of this class
        - self.__class__.name + "_wmc" to the wmc defined by the files for this class
        The second dict maps:
        - "G_loss" to the fuzzy logic loss of this class
        - self.__class__.name to the fuzzy logic loss of this class
        - self.__class__.name + "_wmc" to the wmc defined by the files for this class
        - self.__class__.name + "_wmc_per_sample" to the wmc defined by the files for this class, over each
            sample.
        """
        self.logger.info("Fuzzy logic loss {}".format(self.constraint_name))
        self.logger.info("Importing parser {}".format(self.constraint_name))
        parser = self.__class__._import_sympy(self.experiment, self.constraint_name, self.fuzzy_class)
        g_output_logit = graph_nodes["G_output_logit"]
        self.logger.info("Fuzzy logic loss found G_output_logit of shape {}".format(g_output_logit.shape))

        # set values to probabilities
        toprobs = g_output_logit
        if self.use_sigmoid:
            toprobs = tf.nn.softmax(g_output_logit, name=self.constraint_name + "_to_probabilities", axis=-1)

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

        aggregate = tf.reshape(aggregate, shape=(-1, 2, 1, 3))
        # aggregate: [batch_size * height * width, 2, 1, 3]

        self.logger.info("Fuzzy logic loss reshaping variables to shape {}".format(aggregate.shape))

        """
        Once we have the aggregate [batch_size * (height-1) * width, 16] tensor pass it thought the circuit,
        reduce_prod on constraints related to different rows/cols of the same data sample (as if we were doing
        an AND), then reduce mean over the samples.
        """

        # only use NOT, AND and OR with simple_only=True
        wmc = parser.generate_tf_tree(aggregate, simple_only=True)
        self.logger.info("Fuzzy logic loss wmc of shape {}".format(wmc.shape))
        wmc = tf.reshape(wmc, [-1, (self.sample_height - 1) * self.sample_width])
        self.logger.info("Fuzzy logic wmc reshaped to shape {}".format(wmc.shape))
        wmc_per_sample = tf.reduce_prod(wmc, axis=1)
        self.logger.info("Fuzzy logic loss wmc reduce product to shape {}".format(wmc_per_sample.shape))
        wmc = tf.reduce_mean(wmc_per_sample)
        self.logger.info("Fuzzy logic loss wmc reduced mean to shape {}".format(wmc.shape))
        fuzzy_logic_loss_pre_timing = 1 - wmc
        fuzzy_logic_loss = self._time_fuzzy_loss(graph_nodes, fuzzy_logic_loss_pre_timing)

        # fill dict and return
        nodes = dict()

        # most important key cos it's expected by the trainer
        nodes["G_loss"] = fuzzy_logic_loss
        # adding stuff for statistics
        self._add_FuzzyLogicLoss_statistic_nodes(nodes, fuzzy_logic_loss_pre_timing, fuzzy_logic_loss)
        # add these nodes so that VNU statistics can pick them up later if needed
        self._add_VNU_statistic_nodes(nodes, fuzzy_logic_loss_pre_timing, wmc, wmc_per_sample)
        
        # add these nodes so that they will be logged in terminal
        nodes_to_log = dict()
        self._add_FuzzyLogicLoss_statistic_nodes(nodes_to_log, fuzzy_logic_loss_pre_timing, fuzzy_logic_loss)

        del parser
        return nodes, nodes_to_log, dict()



class FuzzyLogicLoss_cannons(_LevelFuzzyLogicLoss):

    def __init__(self, experiment, *args, **kwargs):
        super().__init__(experiment, *args, **kwargs)
        self.cannons_channels = [self.experiment["TILES_MAP"][x]['index'] for x in ['b', 'B']]
        self.solid_channels = [self.experiment["TILES_MAP"][x]['index'] for x in ['X', 'S', '?', 'Q']]
        self.other_channels = [self.experiment["TILES_MAP"][x]['index'] for x in ['-', 'o', 'b', '<', '>', '[', ']']]

    def _forward(self, **graph_nodes):
        """
        Returns the fuzzy logic loss related to the instance of this class, using the G_output_logit node.
        The sigmoid function is applied by the loss if the property use_sigmoid is set to True.

        :param graph_nodes: Dict of tf graph nodes, G_output_logit must be present.
        :return: A pair of dicts, where the first one is for giving nodes to the trainer, the second is for providing
        nodes that we want to be logged.
        The first dict maps:
        - "G_loss" to the fuzzy logic loss of this class
        - self.__class__.name to the fuzzy logic loss of this class
        - self.__class__.name + "_wmc" to the wmc defined by the files for this class
        The second dict maps:
        - "G_loss" to the fuzzy logic loss of this class
        - self.__class__.name to the fuzzy logic loss of this class
        - self.__class__.name + "_wmc" to the wmc defined by the files for this class
        - self.__class__.name + "_wmc_per_sample" to the wmc defined by the files for this class, over each
            sample.
        """
        self.logger.info(f"Fuzzy logic loss {self.constraint_name}")
        self.logger.info(f"Importing parser {self.constraint_name}")
        parser = self.__class__._import_sympy(self.experiment, self.constraint_name, self.fuzzy_class)
        g_output_logit = graph_nodes["G_output_logit"]
        self.logger.info(f"Fuzzy logic loss found G_output_logit of shape {g_output_logit.shape}")

        # set values to probabilities
        toprobs = g_output_logit
        if self.use_sigmoid:
            toprobs = tf.nn.softmax(g_output_logit, name=self.constraint_name + "_to_probabilities", axis=-1)

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

        aggregate = tf.reshape(aggregate, shape=(-1, 2, 1, 4))
        # aggregate: [batch_size * (height - 1) * width, 2, 1, 4]

        self.logger.info("Fuzzy logic loss reshaping variables to shape {}".format(aggregate.shape))

        """
        Once we have the aggregate [batch_size * (height-1) * width, 8] tensor pass it thought the circuit,
        reduce_prod on constraints related to different rows/cols of the same data sample (as if we were doing
        an AND), then reduce mean over the samples.
        """

        # only use NOT, AND and OR with simple_only=True
        wmc = parser.generate_tf_tree(aggregate, simple_only=True)
        self.logger.info("Fuzzy logic loss wmc of shape {}".format(wmc.shape))
        wmc = tf.reshape(wmc, [-1, (self.sample_height - 1) * self.sample_width])
        self.logger.info("Fuzzy logic wmc reshaped to shape {}".format(wmc.shape))
        wmc_per_sample = tf.reduce_prod(wmc, axis=1)
        self.logger.info("Fuzzy logic loss wmc reduce product to shape {}".format(wmc_per_sample.shape))
        wmc = tf.reduce_mean(wmc_per_sample)
        self.logger.info("Fuzzy logic loss wmc reduced mean to shape {}".format(wmc.shape))
        fuzzy_logic_loss_pre_timing = 1 - wmc
        fuzzy_logic_loss = self._time_fuzzy_loss(graph_nodes, fuzzy_logic_loss_pre_timing)

        # fill dict and return
        nodes = dict()

        # most important key cos it's expected by the trainer
        nodes["G_loss"] = fuzzy_logic_loss
        # adding stuff for statistics
        self._add_FuzzyLogicLoss_statistic_nodes(nodes, fuzzy_logic_loss_pre_timing, fuzzy_logic_loss)
        # add these nodes so that VNU statistics can pick them up later if needed
        self._add_VNU_statistic_nodes(nodes, fuzzy_logic_loss_pre_timing, wmc, wmc_per_sample)
        
        # add these nodes so that they will be logged in terminal
        nodes_to_log = dict()
        self._add_FuzzyLogicLoss_statistic_nodes(nodes_to_log, fuzzy_logic_loss_pre_timing, fuzzy_logic_loss)

        del parser
        return nodes, nodes_to_log, dict()


class FuzzyLogicLoss_pipes(_LevelFuzzyLogicLoss):

    def __init__(self, experiment, *args, **kwargs):
        super().__init__(experiment, *args, **kwargs)

        # retrieve pipe tiles indexes
        self.target_channels = [self.experiment["TILES_MAP"][x]['index'] for x in ['<', '>', '[', ']']]
        self.other_channels = [self.experiment["TILES_MAP"][x]['index'] for x in ['X', 'S', '-', '?', 'Q', 'E', 'o', 'B', 'b']]

    def _forward(self, **graph_nodes):
        """
        Returns the fuzzy logic loss related to the instance of this class, using the G_output_logit node.
        The sigmoid function is applied by the loss if the property use_sigmoid is set to True.

        :param graph_nodes: Dict of tf graph nodes, G_output_logit must be present.
        :return: A pair of dicts, where the first one is for giving nodes to the trainer, the second is for providing
        nodes that we want to be logged.
        The first dict maps:
        - "G_loss" to the fuzzy logic loss of this class
        - self.__class__.name to the fuzzy logic loss of this class
        - self.__class__.name + "_wmc" to the wmc defined by the files for this class
        The second dict maps:
        - "G_loss" to the fuzzy logic loss of this class
        - self.__class__.name to the fuzzy logic loss of this class
        - self.__class__.name + "_wmc" to the wmc defined by the files for this class
        - self.__class__.name + "_wmc_per_sample" to the wmc defined by the files for this class, over each
            sample.
        """
        self.logger.info("Fuzzy logic loss {}".format(self.constraint_name))
        self.logger.info("Importing parser {}".format(self.constraint_name))
        parser = self.__class__._import_sympy(self.experiment, self.constraint_name, self.fuzzy_class)        
        g_output_logit = graph_nodes["G_output_logit"]
        self.logger.info("Fuzzy logic loss found G_output_logit of shape {}".format(g_output_logit.shape))

        # G_output_logit may require normalization
        toprobs = g_output_logit
        self.logger.info("Fuzzy logic loss {} needs normalization: {}".format(self.__class__.__name__, self.use_sigmoid))
        if self.use_sigmoid:
            toprobs = tf.nn.softmax(g_output_logit, axis=-1, name=self.constraint_name + "_to_probabilities")

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
        aggregate = tf.reshape(aggregate, shape=(-1, 2, 2, 5))
        # aggregate has shape: [batch_size * (constraints_per_sample), 2, 2, 5]

        # reshaping to [batch_size * (constraints_per_sample), 20]
        # aggregate = tf.reshape(aggregate, shape=(-1, 20))
        self.logger.info("Fuzzy logic loss reshaping variables to shape {}".format(aggregate.shape))

        """
        Once we have the aggregate [batch_size * (height-1) * width, 16] tensor pass it thought the circuit,
        reduce_prod on constraints related to different rows/cols of the same data sample (as if we were doing
        an AND), then reduce mean over the samples.
        """

        # only use NOT, AND and OR with simple_only=True
        wmc = parser.generate_tf_tree(aggregate, simple_only=True)
        self.logger.info("Fuzzy logic loss wmc of shape {}".format(wmc.shape))
        # now reshaping to [batch_size, constraints_per_sample]
        wmc = tf.reshape(wmc, [-1, (self.sample_height - 1) * (self.sample_width - 1)])
        self.logger.info("Fuzzy logic wmc reshaped to shape {}".format(wmc.shape))
        wmc_per_sample = tf.reduce_prod(wmc, axis=-1)
        self.logger.info("Fuzzy logic loss wmc reduce product to shape {}".format(wmc_per_sample.shape))
        wmc = tf.reduce_mean(wmc_per_sample)
        self.logger.info("Fuzzy logic loss wmc reduced mean to shape {}".format(wmc.shape))
        fuzzy_logic_loss_pre_timing = (1 - wmc) # as in FL paper
        fuzzy_logic_loss = self._time_fuzzy_loss(graph_nodes, fuzzy_logic_loss_pre_timing)

        # fill dict and return
        nodes = dict()

        # most important key cos it's expected by the trainer
        nodes["G_loss"] = fuzzy_logic_loss
        # adding stuff for statistics
        self._add_FuzzyLogicLoss_statistic_nodes(nodes, fuzzy_logic_loss_pre_timing, fuzzy_logic_loss)
        # add these nodes so that VNU statistics can pick them up later if needed
        self._add_VNU_statistic_nodes(nodes, fuzzy_logic_loss_pre_timing, wmc, wmc_per_sample)
        
        # add these nodes so that they will be logged in terminal
        nodes_to_log = dict()
        self._add_FuzzyLogicLoss_statistic_nodes(nodes_to_log, fuzzy_logic_loss_pre_timing, fuzzy_logic_loss)

        del parser
        return nodes, nodes_to_log, dict()


class FuzzyLogicLoss_all_mario(_LevelFuzzyLogicLoss):

    def __init__(self, experiment, *args, **kwargs):
        super().__init__(experiment, *args, **kwargs)

        # retrieve pipe tiles indexes
        self.pipes_tiles = [self.experiment["TILES_MAP"][x]['index'] for x in ['<', '>', '[', ']']]
        self.solid_tiles = [self.experiment["TILES_MAP"][x]['index'] for x in ['X', 'S', '?', 'Q']]
        self.air_tiles = [self.experiment["TILES_MAP"][x]['index'] for x in ['-', 'o']]
        self.cannons_tiles = [self.experiment["TILES_MAP"][x]['index'] for x in ['B', 'b']]
        self.monster_tile = [self.experiment["TILES_MAP"][x]['index'] for x in ['E']]

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
        self.logger.info("Fuzzy logic loss {}".format(self.constraint_name))
        self.logger.info("Importing parser {}".format(self.constraint_name))
        parser = self.__class__._import_sympy(self.experiment, self.constraint_name, self.fuzzy_class)
        g_output_logit = graph_nodes["G_output_logit"]
        self.logger.info("Fuzzy logic loss found G_output_logit of shape {}".format(g_output_logit.shape))

        # G_output_logit may require normalization
        toprobs = g_output_logit
        self.logger.info("Fuzzy logic loss {} needs normalization: {}".format(self.__class__.__name__, self.use_sigmoid))
        if self.use_sigmoid:
            toprobs = tf.nn.softmax(g_output_logit, axis=-1, name=self.constraint_name + "_to_probabilities")

        # order and group channels
        pipes_channels = tf.gather(toprobs, self.pipes_tiles, axis=-1)
        monster_channel = tf.gather(toprobs, self.monster_tile, axis=-1)
        cannons_channels = tf.gather(toprobs, self.cannons_tiles, axis=-1)
        air_tiles = tf.reduce_sum(tf.gather(toprobs, self.air_tiles, axis=-1), axis=-1, keepdims=True)
        solid_tiles = tf.reduce_sum(tf.gather(toprobs, self.solid_tiles, axis=-1), axis=-1, keepdims=True)

        self.logger.info("Fuzzy logic loss found pipes_channels of shape {}".format(pipes_channels.shape))
        self.logger.info("Fuzzy logic loss found monster_channel of shape {}".format(monster_channel.shape))
        self.logger.info("Fuzzy logic loss found cannons_channels of shape {}".format(cannons_channels.shape))
        self.logger.info("Fuzzy logic loss found air_tiles of shape {}".format(air_tiles.shape))
        self.logger.info("Fuzzy logic loss found solid_tiles of shape {}".format(solid_tiles.shape))

        # channels order becomes < | > | [ | ] | E | B | b | XSQE | -o
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
        aggregate = tf.reshape(aggregate, shape=(-1, 2, 2, 9))
        # aggregate has shape: [batch_size * (constraints_per_sample), 2, 2, 9]

        # reshaping to [batch_size * (constraints_per_sample), 36]
        # aggregate = tf.reshape(aggregate, shape=(-1, 36))
        self.logger.info("Fuzzy logic loss reshaping variables to shape {}".format(aggregate.shape))

        """
        Once we have the aggregate [batch_size * (height-1) * width, 36] tensor pass it thought the circuit,
        reduce_prod on constraints related to different rows/cols of the same data sample (as if we were doing
        an AND), then reduce mean over the samples.
        """

        # generate tf tree that encodes fuzzy logic loss
        wmc = parser.generate_tf_tree(aggregate)

        self.logger.info("Fuzzy logic loss wmc of shape {}".format(wmc.shape))
        # now reshaping to [batch_size, constraints_per_sample]
        wmc = tf.reshape(wmc, [-1, (self.sample_height - 1) * (self.sample_width - 1)])
        self.logger.info("Fuzzy logic wmc reshaped to shape {}".format(wmc.shape))
        wmc_per_sample = tf.reduce_prod(wmc, axis=-1)
        self.logger.info("Fuzzy logic loss wmc reduce product to shape {}".format(wmc_per_sample.shape))
        wmc = tf.reduce_mean(wmc_per_sample)
        self.logger.info("Fuzzy logic loss wmc reduced mean to shape {}".format(wmc.shape))
        fuzzy_logic_loss_pre_timing = (1 - wmc) # as in FL paper
        fuzzy_logic_loss = self._time_fuzzy_loss(graph_nodes, fuzzy_logic_loss_pre_timing)

        # fill dict and return
        nodes = dict()

        # fuzzy_logic_loss = tf.Print(fuzzy_logic_loss, [fuzzy_logic_loss], message="passing per the fucking dio fuzzy loss nonostante dovrebbe essere tagliata fora")

        # most important key cos it's expected by the trainer
        nodes["G_loss"] = fuzzy_logic_loss
        # adding stuff for statistics
        self._add_FuzzyLogicLoss_statistic_nodes(nodes, fuzzy_logic_loss_pre_timing, fuzzy_logic_loss)
        # add these nodes so that VNU statistics can pick them up later if needed
        self._add_VNU_statistic_nodes(nodes, fuzzy_logic_loss_pre_timing, wmc, wmc_per_sample)
        
        # add these nodes so that they will be logged in terminal
        nodes_to_log = dict()
        self._add_FuzzyLogicLoss_statistic_nodes(nodes_to_log, fuzzy_logic_loss_pre_timing, fuzzy_logic_loss)

        del parser
        return nodes, nodes_to_log, dict()


"""
Code down here is used to create all constraint classes in an automatic way, adding the fuzzy file
in its relative directory is enough (you don't have to do anything more code wise).
What happens is that for every constraint for which fuzzy files are present, a new class with the name
FuzzyLogic_name of the constraint is created, which inherits from fuzzy logic loss.
"""

def _create_semantic_loss_class(name):
    """ 
    Given a class name, create a class type with that name which inherits from FuzzyLogicLoss
    :param name: Name of the classe to create.
    """
    attributes_dict = {"__init__": lambda self, experiment: _FuzzyLogicLoss.__init__(self, experiment)}
    _tmpclass = type(name, (_FuzzyLogicLoss,), attributes_dict)
    globals()[_tmpclass.__name__] = _tmpclass
    del _tmpclass


"""
For each constraint create a class.
This workaround allows to simply add vtree and sdd files to add a new constraints, without
having to touch up the losses module every time.
"""

_constraints = _get_constraints_names()
for c in _constraints:
    name = "FuzzyLogicLoss_{}".format(c)
    if name not in globals():
        _create_semantic_loss_class(name)
    del name
del _constraints

