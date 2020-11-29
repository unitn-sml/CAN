import numpy as np
import tensorflow as tf
from functools import reduce

from utils import utils_common
from base_layers import Statistic
import semantic_loss.losses as losses
from semantic_loss.util import _get_constraints_names


class SemanticLossStatistics(Statistic):

    def _forward(self, **graph_nodes):
        """
        Add to tensorboard every node which begins with SemanticLoss_

        :param graph_nodes:
        :return: Tf Summaries
        """
        summaries = []
        prefix = "SemanticLoss_"
        with tf.variable_scope("generator_loss"):
            for name, node in graph_nodes.items():
                if name.startswith(prefix) and "per_sample" not in name:
                    summaries.append(tf.summary.scalar(name, node))
        if len(summaries) == 0:
            error_msg = "No summaries to add, this is probably because there are no nodes in graph_nodes beginning " \
                        "with 'SemanticLoss', this statistic should not have been added, assuming this is an " \
                        "erroneous experiment file, exiting. "
            self.experiment["LOGGER"].error(error_msg)
            exit()

        return tf.summary.merge(summaries)


class _WMCStatistic(Statistic):
    """
    Base class for all WMC statistics computed using semantic losses.
    Given the class name it will automatically check if the wmc node already exist in graph_nodes, if it does it
    is returned as a summary, otherwise the semantic loss related to the wmc to log is instantiated, and its
    used internally to compute the wmc for the summary, which is returned.
    """

    def __init__(self, experiment, base_path=""):
        super().__init__(experiment)
        self.constraint_name = "_".join(self.__class__.__name__.split("_")[2:])

    def _forward(self, **graph_nodes):
        """
        Add to tensorboard the wmc from the semantic loss.
        """
        logger = self.experiment["LOGGER"]
        wmc_node_name = "SemanticLoss_" + self.constraint_name + "_wmc"
        logger.info("Doing summary for %s" % wmc_node_name)

        # if its already there just add it to the summaries, otherwise import the sdd and create the tf AC
        if wmc_node_name in graph_nodes:
            logger.info("%s found in graph_nodes, adding it to summaries" % wmc_node_name)
            wmc_node = graph_nodes[wmc_node_name]
            if SemanticLossStatistics in self.experiment["GENERATOR_STATISTICS"]:
                warning_msg = "%s is already present in the summaries, this summary will likely be duplicated, you " \
                              "should check your experiment file." % wmc_node_name
                logger.warning(warning_msg)
        else:
            logger.info("%s not found in graph_nodes, importing sdd and creating TF AC" % wmc_node_name)
            semantic_loss = getattr(losses, "SemanticLoss_%s" % self.constraint_name)(self.experiment)
            nodes, _, _ = semantic_loss(**graph_nodes)
            wmc_node = nodes[wmc_node_name]
            del semantic_loss
        with tf.variable_scope("generator", reuse=True):
            return tf.summary.merge([tf.summary.scalar("WMC_sl_%s" % self.constraint_name, wmc_node)])


class _VNUStatistic(Statistic):
    """
    Base class for all VNU (validity, novelty, uniqueness) statistics computed using semantic losses.
    Validity is computed using the related semantic loss to decide which samples are valid and which ones are not,
    and is equal to the ratio of valid samples to total samples, valid/total.
    Novelty is computed by the ratio of valid samples not in training_data to the total valid samples,
    new valid/total valid.
    Uniqueness is computed by the ratio of unique valid samples to the total valid samples, unique valid/total valid.
    Given the related semantic loss, an instance of this class will add the following statistics to tensorboard:
    - total valid items, and validity
    - total novel items, and novelty
    - total unique items, and uniqueness

    Due to this repository (and HPC) depending on a not really up to date tf version, to compute uniqueness and novelty
    a workaround with quadratic memory requirement has been used. However, whatever the size of the training data,
    the class tries to manage the situation by splitting in batches.
    As long as the cost of memory to compute
    uniqueness can be sustained (quadratic on BATCH_SIZE), the computation of novelty will work as well, given
    that training data is split to be in batches of size [BATCH_SIZE].
    """

    def __init__(self, experiment, base_path=""):
        super().__init__(experiment)
        self.logger = self.experiment["LOGGER"]
        self.constraint_name = "_".join(self.__class__.__name__.split("_")[2:])
        """
        Keep unique training data and split it into batches, splitting is required to not make the memory explode
        when checking for novelty. By using a more up to date version of TF we might be able to process
        everything better in the future, but this is the current workaround.
        """
        self.logger.info("Keeping only unique training data")
        training_data = experiment.training_data
        if experiment["DATASET_TYPE"] != "random_formula":
            training_data = np.unique(training_data, axis=0)
        training_data = np.reshape(training_data, [training_data.shape[0], -1])
        self.logger.info("Splitting unique training data in sub batches")
        training_data = np.array_split(training_data, axis=0,
                                       indices_or_sections=training_data.shape[0] // self.experiment["BATCH_SIZE"])
        self.training_data = [x for x in training_data if x.size > 0]

    def _forward(self, **graph_nodes):
        self.logger.info("Importing semantic loss")
        semantic_loss = getattr(losses, "SemanticLoss_%s" % self.constraint_name)(self.experiment)
        self.logger.info("Building graph gor VNU stats")
        summaries = self._vnu_stats(graph_nodes, semantic_loss)
        return summaries

    def _vnu_stats(self, graph_nodes, semantic_loss):
        """
        :param graph_nodes:
        :param semantic_loss:
        :return:
        """
        """
        Create a "fake" graph_nodes graph in which we substitute G_output_logit, which are raw values, with
        a reshaped version of G_sample, which are actually discretized tensors/samples.
        This is done because the semantic_loss forward method expects/uses G_output_logit as its input node.
        """
        nodes_tmp = {**graph_nodes}
        nodes_tmp.pop("G_output_logit")
        G_sample = tf.reshape(graph_nodes["G_sample"], [-1] + self.experiment["SHAPE"])

        if not self.experiment["HARD_SAMPLING"]:
            # discretize if needed (should be in range[0,1])
            G_sample = G_sample > 0.5
    
        G_sample = tf.cast(G_sample, dtype=tf.float32)

        # sample, do not get all BGAN_SAMPLES from the G_output_logit sample but 1 for
        # each sample of the original of G_output_logit
        if self.experiment["NUM_BGAN_SAMPLES"] is not None:
            self.logger.info("NUM_BGAN_SAMPLES found in experiment, only a sub sample of G_sample will be used to "
                             "compute VNU statistics, one discretized sample for each output distribution of "
                             "G_output_logit")
            indexes = list(range(self.experiment["BATCH_SIZE"]))
            G_sample = tf.gather(G_sample, indexes, axis=0)

        nodes_tmp["G_output_logit"] = G_sample
        # don't use sigmoid on discretized stuff
        semantic_loss.use_sigmoid = False
        nodes, _, _ = semantic_loss(**nodes_tmp)

        wmc_node_name = "SemanticLoss_" + self.constraint_name + "_wmc"
        self.logger.info("Building validity graph")
        validity = nodes[wmc_node_name]
        # indicates if a sample is valid or invalid
        validity_per_sample = nodes[wmc_node_name + "_per_sample"]
        valid_indices = tf.reshape(tf.where(tf.equal(validity_per_sample, 1.0)), [-1])
        valid_samples = tf.gather(G_sample, valid_indices, axis=0)

        # gotta reshape to 2d to perform the uniqueness and novelty
        total_variables = reduce(lambda x, y: x * y, valid_samples.shape[1:])
        valid_samples = tf.reshape(valid_samples, [-1, total_variables])

        self.logger.info("Building uniqueness graph")
        total_unique, uniqueness = _VNUStatistic.uniqueness(valid_samples)
        self.logger.info("Building novelty graph")
        total_novel, novelty = _VNUStatistic.novelty(valid_samples, self.training_data)

        with tf.variable_scope("generator", reuse=True):
            summaries = [tf.summary.scalar("%s Validity" % self.constraint_name, validity),
                         tf.summary.scalar("%s Valid samples" % self.constraint_name, tf.shape(valid_samples)[0]),
                         tf.summary.scalar("%s Uniqueness" % self.constraint_name, uniqueness),
                         tf.summary.scalar("%s Unique samples" % self.constraint_name, total_unique),
                         tf.summary.scalar("%s Novelty" % self.constraint_name, novelty),
                         tf.summary.scalar("%s Novel samples" % self.constraint_name, total_novel)]
            return tf.summary.merge(summaries)

    @staticmethod
    def tf_unique_2d(x):
        """
        Courtesy of https://stackoverflow.com/questions/51487990/find-unique-values-in-a-2d-tensor-using-tensorflow.
        Gotta use this because the TF version we are currently using (and the one in HPC) is not so recent, and lacks
        unique for n-dimensional tensors.
        Had to modify a little because of having unknown batch size.

        :return: Tf op containing the unique elements of x.
        """
        x_shape = tf.shape(x)
        x1 = tf.tile(x, (1, x_shape[0]))  # [[1,2],[1,2],[1,2],[3,4],[3,4],[3,4]..]
        x2 = tf.tile(x, (x_shape[0], 1))  # [[1,2],[1,2],[1,2],[3,4],[3,4],[3,4]..]

        x1_2 = tf.reshape(x1, [x_shape[0] * x_shape[0], x_shape[1]])
        x2_2 = tf.reshape(x2, [x_shape[0] * x_shape[0], x_shape[1]])
        cond = tf.reduce_all(tf.equal(x1_2, x2_2), axis=1)
        cond = tf.reshape(cond, [x_shape[0], x_shape[0]])  # reshaping cond to match x1_2 & x2_2
        cond_shape = tf.shape(cond)
        cond_cast = tf.cast(cond, tf.int32)  # converting condition boolean to int
        cond_zeros = tf.zeros(cond_shape, tf.int32)  # replicating condition tensor into all 0's

        # CREATING RANGE TENSOR
        r = tf.range(x_shape[0])
        r = tf.add(tf.tile(r, [x_shape[0]]), 1)
        r = tf.reshape(r, [x_shape[0], x_shape[0]])

        # converting TRUE=1 FALSE=MAX(index)+1 (which is invalid by default) so when we take min it wont get selected
        # & in end we will only take values <max(indx).
        f1 = tf.multiply(tf.ones(cond_shape, tf.int32), x_shape[0] + 1)
        f2 = tf.ones(cond_shape, tf.int32)
        cond_cast2 = tf.where(tf.equal(cond_cast, cond_zeros), f1, f2)  # if false make it max_index+1 else keep it 1

        # multiply range with new int boolean mask
        r_cond_mul = tf.multiply(r, cond_cast2)
        r_cond_mul2 = tf.reduce_min(r_cond_mul, axis=1)
        r_cond_mul3, unique_idx = tf.unique(r_cond_mul2)
        r_cond_mul4 = tf.subtract(r_cond_mul3, 1)

        # get actual values from unique indexes
        op = tf.gather(x, r_cond_mul4)

        return op

    @staticmethod
    def count_in(x, y):
        """
        Assumes x and y are 2 dimensional, ~ [batch, -1].
        Assumes y elements (axis 1) are unique.
        :param x: 2d Tensor for which we want to know how many elements are in y, each element must be unique.
        :param y: 2d Tensor, each element must be unique.
        :return: A tf node, computes the total number of elements of x which are in y.
        """
        x_shape = tf.shape(x)
        y_shape = tf.shape(y)

        # replicate tensors so that we can compare 1-1 each x element vs each y element
        x_replicated = tf.tile(x, (1, y_shape[0]))
        y_replicated = tf.tile(y, (x_shape[0], 1))
        x_replicated = tf.reshape(x_replicated, [x_shape[0] * y_shape[0], x_shape[1]])
        y_replicated = tf.reshape(y_replicated, [y_shape[0] * x_shape[0], x_shape[1]])

        # element wise equality, followed by equality on all elements
        # x samples which have true on all columns are samples already in y
        equal = tf.equal(x_replicated, y_replicated)
        equal_all_dims = tf.reduce_all(equal, axis=1)

        total_already_in = tf.reduce_sum(tf.cast(equal_all_dims, tf.int32))
        return total_already_in

    @staticmethod
    def novelty(x, data):
        """
        Returns the tf nodes for the total novel items (x not in data) and the novelty (total x not in data / total x).

        :param x: Tensor of 2d shape, [batch, -1], each element on axis 0 is a sample.
        :param data: List of tensors of 2d shape, [batch, -1], each element on axis 0 is a sample.
        :return: Tf nodes for total novel items and novelty
        """

        total_x = tf.shape(x)[0]
        total_x_in_y = 0

        # for each batch of data check che number of hits from samples in x
        for data_batch in data:
            count_x_in_y = _VNUStatistic.count_in(x, data_batch)
            total_x_in_y = total_x_in_y + count_x_in_y

        count_not_x_in_y = total_x - total_x_in_y
        novelty = tf.cast(count_not_x_in_y, tf.float32) / tf.cast(total_x, tf.float32)
        novelty = tf.cond(tf.is_nan(novelty), true_fn=lambda: tf.constant(0.), false_fn=lambda: novelty)
        return count_not_x_in_y, novelty

    @staticmethod
    def uniqueness(x):
        """
        Returns the tf nodes for the total unique items and uniqueness (unique items / total items).

        :param x: Tensor of 2d shape, [batch, -1], each element on axis 0 is a sample.
        :return: Tf nodes for total unique items and uniqueness.
        """
        unique_valid_samples = _VNUStatistic.tf_unique_2d(x)

        total_unique = tf.shape(unique_valid_samples)[0]
        total_valid = tf.shape(x)[0]
        tu_float = tf.cast(total_unique, tf.float32)
        tv_float = tf.cast(total_valid, tf.float32)
        uniqueness = tu_float / tv_float

        uniqueness = tf.cond(tf.is_nan(uniqueness), true_fn=lambda: tf.constant(0.), false_fn=lambda: uniqueness)
        return total_unique, uniqueness


##############################################
#############################################
"""
For each semantic loss class create a related WMC statistics and VNU statistics class, automatically.
"""


def _create_wmc_class(name):
    """
    Given a class name, create a class type with that name which inherits from WMCStatistic
    :param name: Name of the classe to create.
    """
    attributes_dict = {"__init__": lambda self, experiment: _WMCStatistic.__init__(self, experiment)}
    _tmpclass = type(name, (_WMCStatistic,), attributes_dict)
    globals()[_tmpclass.__name__] = _tmpclass
    del _tmpclass


def _create_vnu_class(name):
    """
    Given a class name, create a class type with that name which inherits from WMCStatistic
    :param name: Name of the classe to create.
    """
    attributes_dict = {"__init__": lambda self, experiment: _VNUStatistic.__init__(self, experiment)}
    _tmpclass = type(name, (_VNUStatistic,), attributes_dict)
    globals()[_tmpclass.__name__] = _tmpclass
    del _tmpclass


for sem_loss_name in utils_common.get_module_classes(losses.__name__):
    constraint_name = "_".join(sem_loss_name.split("_")[1:])
    name = "WMC_sl_{}".format(constraint_name)
    if name not in globals():
        _create_wmc_class(name)

    name = "VNU_sl_{}".format(constraint_name)
    if name not in globals():
        _create_vnu_class(name)
    del name
