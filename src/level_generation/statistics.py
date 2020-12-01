import tensorflow as tf
from base_layers import Statistic
import numpy as np
from level_generation.reachability import find_reachable
import json, os

class _LevelStatistic(Statistic):
    """
    Generate classic metrics for a given node.
    Standard metrics include: mean, standard deviation, max and min.
    An histogram of the target node is also passed to tensorboard
    """
    def __init__(self, experiment):
        super().__init__(experiment)
        self.height, self.width, self.channels = experiment["SHAPE"] 

    def _stats_summaries(self, node, prefix=""):
        mean = tf.reduce_mean(node)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(node - mean)))
        maxx = tf.reduce_max(node)
        minn = tf.reduce_min(node)

        mean_summ = tf.summary.scalar(prefix + "mean", mean)
        stddev_summ = tf.summary.scalar(prefix + "stddev", stddev)
        max_summ = tf.summary.scalar(prefix + "max", maxx)
        min_summ = tf.summary.scalar(prefix + "min", minn)
        histogram_summ = tf.summary.histogram(prefix + "histogram", node)

        return tf.summary.merge([mean_summ, stddev_summ, max_summ, min_summ, histogram_summ])


########################################################################################################################
######################################## Discriminator Statistics ######################################################
########################################################################################################################

class LevelDiscriminatorStatistics(_LevelStatistic):
    """
    Base class for print statistics on the discriminator output.
    Requires both D_fake and D_real to extract info about them.
    """

    def _pre_processing(self, **graph_nodes):
        self._requires(["D_real", "D_fake"], **graph_nodes)

    def _forward(self, **graph_nodes):
        D_real = graph_nodes["D_real"]
        D_fake = graph_nodes["D_fake"]

        with tf.variable_scope("discriminator_real"):
            with tf.variable_scope("out"):
                summaries_real = self._stats_summaries(D_real)

        with tf.variable_scope("discriminator_fake"):
            with tf.variable_scope("out"):
                summaries_fake = self._stats_summaries(D_fake)

        return tf.summary.merge([summaries_real, summaries_fake])


########################################################################################################################
######################################## Generator Statistics ##########################################################
########################################################################################################################

class LevelGeneratorStatistics(_LevelStatistic):
    """
    Base class for print statistics on the generator output logits.
    Requires G_sample_logit to extract info about it.
    """

    def _pre_processing(self, **graph_nodes):
        self._requires("G_output_logit", **graph_nodes)

    def _forward(self, **graph_nodes):
        g_out = graph_nodes["G_output_logit"]
        shape = g_out.shape.as_list()

        with tf.variable_scope("generator"):
            with tf.variable_scope("out"):
                raw_stats_summaries = [
                    self._stats_summaries(tf.reshape(g_out[:, :, :, i], [-1, *shape[1:3], 1]), "ch_{0}".format(i)) 
                    for i in range(shape[-1])
                ]
        return tf.summary.merge(raw_stats_summaries)


########################################################################################################################
###################################### Discriminator Loss Statistics ###################################################
########################################################################################################################

class LevelDiscriminatorLossStatistics(_LevelStatistic):
    """
    Base class for print statistics on the discriminator loss.
    Requires D_loss to extract info about it.
    """

    def _pre_processing(self, **graph_nodes):
        self._requires("D_loss", **graph_nodes)

    def _forward(self, **graph_nodes):
        D_loss = graph_nodes["D_loss"]

        with tf.variable_scope("discriminator_loss"):
            loss_summary = tf.summary.scalar("batch-by-batch", D_loss)

        return tf.summary.merge([loss_summary])


########################################################################################################################
######################################## Generator Loss Statistics #####################################################
########################################################################################################################

class LevelGeneratorLossStatistics(_LevelStatistic):
    """
    Base class for print statistics on the generator loss.
    Requires G_loss to extract info about it.
    """

    def _pre_processing(self, **graph_nodes):
        self._requires("G_loss", **graph_nodes)

    def _forward(self, **graph_nodes):
        G_loss = graph_nodes["G_loss"]

        with tf.variable_scope("generator_loss"):
            loss_summary = tf.summary.scalar("batch-by-batch", G_loss)

        return tf.summary.merge([loss_summary])


########################################################################################################################
###################################### Constraints Satisfaction Statistics #############################################
########################################################################################################################


class LevelGeneratorSemanticLossStatistics(_LevelStatistic):
    """
    Search the dict of important nodes graph_nodes for keys starting with G_semantic_loss_original_ or
    G_semantic_loss_timed_ and send their value to tensorboard under metrics/{key}
    """

    def _forward(self, **graph_nodes):
        res = []
        for key, value in graph_nodes.items():
            if key.startswith("G_semantic_loss_original_") or key.startswith("G_semantic_loss_timed_"):
                with tf.variable_scope("metrics", reuse=True):
                    res.append(tf.summary.scalar(key, value))
        return tf.summary.merge(res) if len(res) > 0 else None


class LevelGeneratorFuzzyLogicLossStatistics(_LevelStatistic):
    """
    Search the dict of important nodes graph_nodes for keys starting with G_fuzzy_logic_loss_original_ or
    G_fuzzy_logic_loss_timed_ and send their value to tensorboard under metrics/{key}
    """

    def _forward(self, **graph_nodes):
        res = []
        for key, value in graph_nodes.items():
            if key.startswith("G_fuzzy_logic_loss_original_") or key.startswith("G_fuzzy_logic_loss_timed_"):
                with tf.variable_scope("metrics", reuse=True):
                    res.append(tf.summary.scalar(key, value))
        return tf.summary.merge(res) if len(res) > 0 else None


###### REACHABILITY

class LevelGeneratorAlgorithmicReachabilityStatistics(_LevelStatistic):
    """
    Use G_sample node of discrete samples to extract and generate statistics about the reachability, in the same
    way it is defined by SemanticLoss_reachability
    """

    def __init__(self, experiment):
        super().__init__(experiment)
        self.solid = [self.experiment["TILES_MAP"][x]['index'] for x in ['X', 'S', '?', 'Q', '[', ']', '<', '>', 'B', 'b']]

    def _pre_processing(self, **graph_nodes):
        self._requires("G_sample", **graph_nodes)
        # loading jumps file before the _forward method
        settings = self.experiment["SETTINGS"]
        self.jumps = settings['jumps']

    def _forward(self, **graph_nodes):
        G_sample = graph_nodes["G_sample"]
        # summing up solid channels, remember: G_sample is already onehot encoded
        G_sample = tf.reduce_sum(tf.gather(G_sample, self.solid, axis=-1), axis=-1)
        G_sample = tf.cast(G_sample, tf.bool)
        
        metric = tf.py_func(
            self._batch_reachability,
            [G_sample],
            tf.float32,
            name='reachability_algorithm_op'
        )

        with tf.variable_scope("metrics", reuse=True):
            return tf.summary.scalar("reachability_algorithm", metric)

    def _batch_reachability(self, x):
        # x.shape = (None, height, width)
        reachabilities = []
        for sample in x:
            sample = find_reachable(sample, self.jumps)
            # consider 3x2 rectangles bottom left and right
            sample = self._reachability(sample).astype(np.float32)
            reachabilities.append(sample)
        return np.mean(reachabilities)

    def _reachability(self, sample):
        return (sample[-2, 0] * (1 - sample[-1, 0])) * (sample[-2, -1] * (1 - sample[-1, -1]))


class LevelGeneratorPlayabilityStatistics(_LevelStatistic):
    """
    Use G_sample node of discrete samples to extract and generate statistics about the reachability, in the same
    way it is defined by SemanticLoss_reachability
    """

    def __init__(self, experiment):
        super().__init__(experiment)
        self.solid = [self.experiment["TILES_MAP"][x]['index'] for x in ['X', 'S', '?', 'Q', '[', ']', '<', '>', 'B', 'b']]

    def _pre_processing(self, **graph_nodes):
        self._requires("G_sample", **graph_nodes)
        # loading jumps file before the _forward method
        settings = self.experiment["SETTINGS"]
        self.jumps = settings['jumps']

    def _forward(self, **graph_nodes):
        G_sample = graph_nodes["G_sample"]
        # summing up solid channels, remember: G_sample is already onehot encoded
        G_sample = tf.reduce_sum(tf.gather(G_sample, self.solid, axis=-1), axis=-1)
        G_sample = tf.cast(G_sample, tf.bool)
        
        metric = tf.py_func(
            self._batch_reachability,
            [G_sample],
            tf.float32,
            name='reachability_algorithm_op'
        )

        with tf.variable_scope("metrics", reuse=True):
            return tf.summary.scalar("playability", metric)

    def _batch_reachability(self, x):
        # x.shape = (None, height, width)
        reachabilities = []
        for sample in x:
            sample = find_reachable(sample, self.jumps)
            # consider 3x2 rectangles bottom left and right
            sample = self._reachability(sample).astype(np.float32)
            reachabilities.append(sample)
        return np.mean(reachabilities)

    def _reachability(self, sample):
        sample = sample.astype(bool)
        return np.any(sample[:, -1])
         
'''
class LevelGeneratorAlgorithmicReachabilityExtendedStatistics(LevelGeneratorAlgorithmicReachabilityStatistics):

    def _reachability(self, sample):
        return np.product((np.sum(sample, axis=0) >= 1).astype(int))


class LevelGeneratorAlgorithmicReachabilitySimpleStatistics(LevelGeneratorAlgorithmicReachabilityStatistics):

    def _reachability(self, sample):
        return int(np.sum(sample[10:12, 2:4]) + np.sum(sample[6:8, 6:8]) == 8)
'''

class LevelGeneratorReachabilityCNNStatistics(_LevelStatistic):
    """
    Extract and print some statistics about the reachability CNN used by SemanticLoss_rechability class.
    The reachability CNN is created using the reachability_network.py tool.
    """

    def _pre_processing(self, **graph_nodes):
        self._requires("reachability_aggregate", **graph_nodes)
        
    def _forward(self, **graph_nodes):
        reachability_aggregate = graph_nodes["reachability_aggregate"]
        
        with tf.variable_scope("metrics", reuse=True):
            with tf.variable_scope("reachability_aggregate", reuse=True):
                return self._stats_summaries(reachability_aggregate)



########## PIPES

class LevelGeneratorPipesPerfectionStatistic(_LevelStatistic):
    """
    Take G_sample and compute the percentage of perfect objects as they are defined in SemanticLoss_pipes class.
    This metric is computed on discrete date but its value is really similar to the weighted model count of 
    SemanticLoss_pipes.
    """

    def __init__(self, experiment):
        super().__init__(experiment)
        self.target_channels = [self.experiment["TILES_MAP"][x]['index'] for x in ['<', '>', '[', ']']]

    def _pre_processing(self, **graph_nodes):
        self._requires("G_sample", **graph_nodes)

    def _forward(self, **graph_nodes):
        G_sample = graph_nodes["G_sample"]

        sampler = tf.gather(G_sample, self.target_channels, axis=-1)
        
        metric = tf.py_func(
            self._are_perfects,
            [sampler],
            tf.float32,
            name='reachability_algorithm_op'
        )
        
        with tf.variable_scope("metrics", reuse=True):
            return tf.summary.scalar("pipes_perfection", metric)

    def _are_perfects(self, samples):
        res = []
        for sample in samples:
            res.append(self._is_perfect_sample(sample))
        return np.mean(res).astype(np.float32)

    def _is_perfect_sample(self, sample):
        sample = np.pad(sample, [[1, 1], [1, 1], [0, 0]], mode='constant', constant_values=0)
        for i in range(sample.shape[0] - 1):
            for j in range(sample.shape[1] - 1):
                if not self._is_perfect_square(sample[i:i+2, j:j+2, :]):
                    return 0
        return 1

    def _is_perfect_square(self, x):
        x = x.astype(bool)
        return \
            (x[0][0][0] == x[0][1][1]) and \
            (x[1][0][0] == x[1][1][1]) and \
            True and \
            (x[0][0][2] == x[0][1][3]) and \
            (x[1][0][2] == x[1][1][3]) and \
            True and \
            (not x[0][0][0] or x[1][0][2]) and \
            (not x[0][1][1] or x[1][1][3]) and \
            True and \
            (not x[1][0][2] or (x[0][0][2] or x[0][0][0])) and \
            (not x[1][1][3] or (x[0][1][3] or x[0][1][1]))


class LevelGeneratorPipesNumberStatistic(_LevelStatistic):
    """
    Statistic about the total expected number of tiles of type pipes for each level.
    """

    def __init__(self, experiment):
        super().__init__(experiment)
        self.pipes_channels = [self.experiment["TILES_MAP"][x]['index'] for x in ['[', ']', '<', '>']]

    def _pre_processing(self, **graph_nodes):
        #self._requires(["G_sample", "PipesNumberLoss"], **graph_nodes)
        self._requires("G_sample", **graph_nodes)

    def _forward(self, **graph_nodes):
        G_sample = graph_nodes["G_sample"]
        #PipesNumberLoss = graph_nodes["PipesNumberLoss"]

        sampler = tf.gather(G_sample, self.pipes_channels, axis=-1)
        # sum over each sample and take the average over all samples
        metric = tf.reduce_sum(sampler, axis=[-3, -2, -1])
        metric = tf.cast(metric, tf.float32)
        metric = tf.reduce_mean(metric)

        with tf.variable_scope("metrics", reuse=True):
            scalar_1 = tf.summary.scalar("pipes_number", metric)
            #scalar_2 = tf.summary.scalar("n_pipes_loss", PipesNumberLoss)
            #return tf.summary.merge([scalar_1, scalar_2])
            return tf.summary.merge([scalar_1])


############# ONEHOT

class LevelGeneratorOnehotStatistic(_LevelStatistic):
    """
    Measure the decision of the generator on each tile type of each pixel.
    In particular, the average standard deviation of the last channels is given
    and the average difference between the most probable and the second most probable
    tile type.
    """

    def _pre_processing(self, **graph_nodes):
        self._requires("G_output_logit", **graph_nodes)

    def _forward(self, **graph_nodes):
        G_output_logit = graph_nodes["G_output_logit"]
        
        first_to_second_distance = tf.py_func(
            self._first_to_second_distance,
            [G_output_logit],
            tf.float32
        )
        standard_dev = tf.py_func(
            self._standard_dev,
            [G_output_logit],
            tf.float32
        )
        
        with tf.variable_scope("metrics", reuse=True):
            f2sd = tf.summary.scalar("first_to_second_distance", first_to_second_distance)
            stddev = tf.summary.scalar("average_standard_dev_channel", standard_dev)
            return tf.summary.merge([f2sd, stddev])

    def _first_to_second_distance(self, samples):
        samples = np.sort(samples, axis=-1)
        # best - second best
        res = samples[:, :, :, -1] - samples[:, :, :, -2]
        return np.mean(res).astype(np.float32)

    def _standard_dev(self, samples):
        res = np.std(samples, axis=-1)
        return np.mean(res).astype(np.float32)


####### CANNONS

class LevelGeneratorCannonsStatistic(_LevelStatistic):
    """
    Measure the decision of the generator on each tile type of each pixel.
    In particular, the average standard deviation of the last channels is given
    and the average difference between the most probable and the second most probable
    tile type.
    """

    def __init__(self, experiment):
        super().__init__(experiment)
        self.cannons_channels = [self.experiment["TILES_MAP"][x]['index'] for x in ['b', 'B']]

    def _pre_processing(self, **graph_nodes):
        self._requires("G_sample", **graph_nodes)
        #self._requires("CannonsNumberLoss", **graph_nodes)

    def _forward(self, **graph_nodes):
        G_sample = graph_nodes["G_sample"]
        #CannonsNumberLoss = graph_nodes["CannonsNumberLoss"]

        cannons_tile_number = tf.gather(G_sample, self.cannons_channels, axis=-1)
        cannons_tile_number = tf.reduce_sum(cannons_tile_number, axis=[-3, -2, -1])
        cannons_tile_number = tf.cast(cannons_tile_number, tf.float32)
        cannons_tile_number = tf.reduce_mean(cannons_tile_number, axis=0)
        
        with tf.variable_scope("metrics", reuse=True):
            scalar_1 = tf.summary.scalar("cannons_number", cannons_tile_number)
            #scalar_2 = tf.summary.scalar("n_cannons_loss", CannonsNumberLoss)
            return tf.summary.merge([scalar_1])


####### MONSTERS

class LevelGeneratorMonstersStatistic(_LevelStatistic):
    """
    Measure the decision of the generator on each tile type of each pixel.
    In particular, the average standard deviation of the last channels is given
    and the average difference between the most probable and the second most probable
    tile type.
    """

    def __init__(self, experiment):
        super().__init__(experiment)
        self.monsters_channels = self.experiment["TILES_MAP"]['E']['index']

    def _pre_processing(self, **graph_nodes):
        self._requires("G_sample", **graph_nodes)

    def _forward(self, **graph_nodes):
        G_sample = graph_nodes["G_sample"]

        monsters_tile_number = G_sample[:, :, :, self.monsters_channels]
        monsters_tile_number = tf.reduce_sum(monsters_tile_number, axis=[-2, -1])
        monsters_tile_number = tf.cast(monsters_tile_number, tf.float32)
        monsters_tile_number = tf.reduce_mean(monsters_tile_number, axis=0)
        
        with tf.variable_scope("metrics", reuse=True):
            return tf.summary.scalar("monsters_tile_number", monsters_tile_number)


####### JS_batch

class LevelGeneratorJSStatistic(_LevelStatistic):
    """
    """

    def _pre_processing(self, **graph_nodes):
        self._requires("JS_loss", **graph_nodes)

    def _forward(self, **graph_nodes):
        JS_loss = graph_nodes["JS_loss"]

        with tf.variable_scope("metrics", reuse=True):
            return tf.summary.scalar("JS_loss", JS_loss)

####### L1 Norm

"""
class LevelGeneratorL1NormStatistic(_LevelStatistic):

    def _pre_processing(self, **graph_nodes):
        self._requires("l1_norm", **graph_nodes)

    def _forward(self, **graph_nodes):
        l1_norm = graph_nodes["l1_norm"]

        with tf.variable_scope("metrics", reuse=True):
            return tf.summary.scalar("L1-norm", l1_norm)
"""

class LevelGeneratorL1NormStatistic(_LevelStatistic):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_height, self.sample_width, self.sample_channels = self.experiment["SHAPE"]

    def _pre_processing(self, **graph_nodes):
        self._requires("G_output_logit", **graph_nodes)

    def _forward(self, **graph_nodes):

        # retrieve G_output_logit
        g_output_logit = graph_nodes["G_output_logit"]
        # shape: [bs, h, w, n_channels]

        # softmax only on channel axis
        probs = tf.nn.softmax(g_output_logit, axis=-1)

        # batch has size [bs, h, w, ch] -> reshaping to [bs, -1]
        probs = tf.reshape(probs, shape=[-1, self.sample_height * self.sample_width * self.sample_channels])
        # now each element is a vector, nothing else changed

        # expanding to have two tensors with shape [1, bs, *] and [bs, 1, *] and allow broadcasting
        probs_a = tf.expand_dims(probs, 0)
        probs_b = tf.expand_dims(probs, 1)

        l1_norm = tf.reduce_mean(tf.abs(probs_a - probs_b)) / 2

        with tf.variable_scope("metrics", reuse=True):
            return tf.summary.scalar("L1-norm", l1_norm)