from base_layers import Statistic
import tensorflow as tf


class _PolygonsFloorStatistics(Statistic):

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

class GanDiscriminatorStatistics(_PolygonsFloorStatistics):

    def _pre_processing(self, **graph_nodes):
        assert "D_real" in graph_nodes
        assert "D_fake" in graph_nodes

    def _forward(self, **graph_nodes):
        D_real = graph_nodes["D_real"]
        D_fake = graph_nodes["D_fake"]

        with tf.variable_scope("discriminator_real"):
            with tf.variable_scope("out"):
                summaries = self._stats_summaries(D_real)

        with tf.variable_scope("discriminator_fake"):
            with tf.variable_scope("out"):
                tf.summary.merge([summaries, self._stats_summaries(D_fake)])

        return summaries


class CanDiscriminatorStatistics(_PolygonsFloorStatistics):

    def _pre_processing(self, **graph_nodes):
        assert "D_real" in graph_nodes
        assert "D_fake" in graph_nodes

    def _forward(self, **graph_nodes):
        D_real = graph_nodes["D_real"]
        D_fake = graph_nodes["D_fake"]

        with tf.variable_scope("discriminator_real"):
            with tf.variable_scope("out"):
                summaries = tf.summary.merge([self._stats_summaries(D_real)])

            with tf.variable_scope("hidden4"):
                if "D_hidden4_real" in graph_nodes:
                    summaries = tf.summary.merge([summaries, self._stats_summaries(graph_nodes["D_hidden4_real"])])

            with tf.variable_scope("hidden4_sigmoid"):
                if "D_hidden4_sigmoid_real" in graph_nodes:
                    summaries = tf.summary.merge([summaries,
                                                  self._stats_summaries(graph_nodes["D_hidden4_sigmoid_real"])])

            with tf.variable_scope("D_out_kernel"):
                if "D_out_kernel_real" in graph_nodes:
                    summaries = tf.summary.merge([summaries, self._stats_summaries(graph_nodes["D_out_kernel_real"])])

            with tf.variable_scope("D_constraints_kernel"):
                if "D_constraints_kernel_real" in graph_nodes:
                    summaries = \
                        tf.summary.merge([summaries, self._stats_summaries(graph_nodes["D_constraints_kernel_real"])])

        with tf.variable_scope("discriminator_fake"):
            with tf.variable_scope("out"):
                summaries = tf.summary.merge([summaries, self._stats_summaries(D_fake)])

            with tf.variable_scope("hidden4"):
                if "D_hidden4_fake" in graph_nodes:
                    summaries = tf.summary.merge([summaries, self._stats_summaries(graph_nodes["D_hidden4_fake"])])

            with tf.variable_scope("hidden4_sigmoid"):
                if "D_hidden4_sigmoid_fake" in graph_nodes:
                    summaries = tf.summary.merge([summaries,
                                                  self._stats_summaries(graph_nodes["D_hidden4_sigmoid_fake"])])

            with tf.variable_scope("D_out_kernel"):
                if "D_out_kernel_fake" in graph_nodes:
                    summaries = tf.summary.merge([summaries, self._stats_summaries(graph_nodes["D_out_kernel_fake"])])

            with tf.variable_scope("D_constraints_kernel"):
                if "D_constraints_kernel_fake" in graph_nodes:
                    summaries = \
                        tf.summary.merge([summaries, self._stats_summaries(graph_nodes["D_constraints_kernel_fake"])])

        return summaries


########################################################################################################################
######################################## Generator Statistics ##########################################################
########################################################################################################################

class BinomialGeneratorStatistics(_PolygonsFloorStatistics):

    def _pre_processing(self, **graph_nodes):
        assert "G_output_logit" in graph_nodes

    def _forward(self, **graph_nodes):
        g_out = graph_nodes["G_output_logit"]

        with tf.variable_scope("generator/out"):
            raw_stats_summaries = self._stats_summaries(g_out, "raw-")
        return raw_stats_summaries


class MultinomialGeneratorStatistics(_PolygonsFloorStatistics):

    def _pre_processing(self, **graph_nodes):
        assert "G_output_logit" in graph_nodes

    def _forward(self, **graph_nodes):
        g_out = graph_nodes["G_output_logit"]

        with tf.variable_scope("generator/out"):
            raw_stats_summaries = [self._stats_summaries(tf.reshape(g_out[:, :, :, i],
                                                                    [-1, *self.experiment["SHAPE"][:2], 1]),
                                                         "raw-" + str(i)) for i in range(self.experiment["SHAPE"][-1])]
        return raw_stats_summaries


########################################################################################################################
###################################### Discriminator Loss Statistics ###################################################
########################################################################################################################

class DiscriminatorLossStatistics(_PolygonsFloorStatistics):

    def _pre_processing(self, **graph_nodes):
        assert "D_loss" in graph_nodes
        assert "D_p_real" in graph_nodes

    def _forward(self, **graph_nodes):
        D_loss = graph_nodes["D_loss"]
        D_adversarial_loss = graph_nodes["D_adversarial_loss"]
        p_real = graph_nodes["D_p_real"]

        with tf.variable_scope("discriminator_loss"):
            loss_summary = tf.summary.scalar("batch-by-batch", D_loss)
            adv_summary = tf.summary.scalar("D_adversarial", D_adversarial_loss)
            p_real_summary = tf.summary.scalar("p-real-1", p_real)

        return tf.summary.merge([loss_summary, p_real_summary, adv_summary])


class DiscriminatorLogitsStatistics(_PolygonsFloorStatistics):

    def _pre_processing(self, **graph_nodes):
        assert "D_loss" in graph_nodes
        assert "D_p_real" in graph_nodes

    def _forward(self, **graph_nodes):
        D_real = graph_nodes["D_real"]
        D_fake = graph_nodes["D_fake"]

        with tf.variable_scope("discriminator_loss"):
            mreal = tf.summary.scalar("mean_D(real)", tf.math.reduce_mean(D_real))
            mfake = tf.summary.scalar("mean_D(fake)", tf.math.reduce_mean(D_fake))

        return tf.summary.merge([mreal, mfake])


########################################################################################################################
######################################## Generator Loss Statistics #####################################################
########################################################################################################################

class GeneratorLossStatistics(_PolygonsFloorStatistics):

    def _pre_processing(self, **graph_nodes):
        assert "G_loss" in graph_nodes
        assert "G_p_fake" in graph_nodes

    def _forward(self, **graph_nodes):
        G_loss = graph_nodes["G_loss"]
        G_adversarial_loss = graph_nodes["G_adversarial_loss"]
        p_fake = graph_nodes["G_p_fake"]

        with tf.variable_scope("generator_loss"):
            loss_summary = tf.summary.scalar("batch-by-batch", G_loss)
            adv_summary = tf.summary.scalar("G_adversarial", G_adversarial_loss)
            p_fake_summary = tf.summary.scalar("p-fake-0", p_fake)

        return tf.summary.merge([loss_summary, p_fake_summary, adv_summary])
