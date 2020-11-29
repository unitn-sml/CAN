"""
This file contains the common losses currently available, for discriminators
and generators.
Functions defined here usually accept a series of arguments (i.e. the real and fake data tensors
for the discriminator), and return a tensor representing the loss, some tf/tb summaries and
a dict mapping names of a value to a tensor, to help logging data.
! get_losses is the only public function that does not serve this purpose, serving instead as a
entry point for the trainer class to retrieve whatever loss was specified in the experiment.
"""
import tensorflow as tf
from base_layers import Loss

from utils.utils_tf import pad_left


########################################################################################################################
############################################## DISCRIMINATOR LOSSES ####################################################
########################################################################################################################

class _PolygonsFloorDiscriminatorLoss(Loss):

    def _pre_processing(self, **graph_nodes):
        assert "D_real" in graph_nodes
        assert "D_fake" in graph_nodes


class BganDiscriminatorLoss(_PolygonsFloorDiscriminatorLoss):

    def _forward(self, **graph_nodes):
        D_real = graph_nodes["D_real"]
        D_fake = graph_nodes["D_fake"]
        D_loss = tf.reduce_mean(tf.nn.softplus(-D_real)) + tf.reduce_mean(tf.nn.softplus(-D_fake)) + \
            tf.reduce_mean(D_fake)

        cname = self.__class__.__name__
        with tf.name_scope(cname):
            p_real = tf.reduce_mean(tf.sigmoid(D_real))

        d_results = {
            'D_loss': D_loss,
            'p(real==1)': p_real,
            'p(fake==1)': tf.reduce_mean(tf.sigmoid(D_fake)),
            'E[logit(real)]': tf.reduce_mean(D_real),
        }
        d_results = {cname + "_" + key: val for key, val in d_results.items()}

        res = dict()
        res["D_adversarial_loss"] = D_loss
        res["D_loss"] = D_loss
        res["D_p_real"] = p_real

        return res, d_results, dict()


BganDiscriminatorLossMulti = BganDiscriminatorLoss


########################################################################################################################
############################################## GENERATOR LOSSES ########################################################
########################################################################################################################

class _PolygonsFloorGeneratorLoss(Loss):

    def _pre_processing(self, **graph_nodes):
        assert "G_sample" in graph_nodes
        assert "D_fake" in graph_nodes
        assert "G_output_logit" in graph_nodes


class BganGeneratorLoss(_PolygonsFloorGeneratorLoss):

    def _forward(self, **graph_nodes):
        D_fake = graph_nodes["D_fake"]
        g_output_logit = graph_nodes["G_output_logit"]
        G_sample = graph_nodes["G_sample"]

        log_w = tf.reshape(D_fake, [self.experiment["NUM_BGAN_SAMPLES"], self.experiment["BATCH_SIZE"]])  # (20, 64)
        # notes.pdf, Appendix B, penultimate calculation
        log_g = -tf.reduce_sum((1. - G_sample) * pad_left(g_output_logit) + pad_left(tf.nn.softplus(-g_output_logit)),
                               axis=[2, 3, 4])  # (20, 32)
        log_N = tf.log(tf.cast(log_w.shape[0], dtype=tf.float32))  # ()
        log_Z_est = tf.reduce_logsumexp(log_w - log_N, axis=0)  # (64,)
        log_w_tilde = log_w - pad_left(log_Z_est) - log_N  # (20, 64)
        w_tilde = tf.exp(log_w_tilde)  # (20, 64)

        G_loss = -tf.reduce_mean(tf.reduce_sum(w_tilde * log_g, 0))
        cname = self.__class__.__name__
        with tf.name_scope(cname):
            p_fake = 1 - tf.reduce_mean(tf.sigmoid(D_fake))

        g_results = {
            'g_loss': G_loss,
            'p(fake==0)': p_fake,
            'p(fake)': tf.reduce_mean(tf.sigmoid(D_fake)),
            'E[logit(fake)]': tf.reduce_mean(D_fake),
        }
        g_results = {cname + "_" + key: val for key, val in g_results.items()}

        res = dict()
        res["G_adversarial_loss"] = G_loss
        res["G_loss"] = G_loss
        res["G_p_fake"] = p_fake

        return res, g_results, dict()


class BganGeneratorLossMulti(_PolygonsFloorGeneratorLoss):

    def _forward(self, **graph_nodes):
        D_fake = graph_nodes["D_fake"]
        g_output_logit = graph_nodes["G_output_logit"]
        G_sample = graph_nodes["G_sample"]

        log_w = tf.reshape(D_fake, [
                           self.experiment["NUM_BGAN_SAMPLES"], self.experiment["BATCH_SIZE"]])  # (20, 64)
        # notes.pdf, Appendix B, penultimate calculation
        log_g = tf.reduce_sum((G_sample * (g_output_logit - tf.reduce_logsumexp(
            g_output_logit, axis=3, keep_dims=True
        ))[None, :, :, :, :]), axis=[2, 3, 4])

        log_N = tf.log(tf.cast(log_w.shape[0], dtype=tf.float32))  # ()
        log_Z_est = tf.reduce_logsumexp(log_w - log_N, axis=0)  # (64,)
        log_w_tilde = log_w - pad_left(log_Z_est) - log_N  # (20, 64)
        w_tilde = tf.exp(log_w_tilde)  # (20, 64)

        G_loss = -tf.reduce_mean(tf.reduce_sum(w_tilde * log_g, 0))
        cname = self.__class__.__name__
        with tf.name_scope(cname):
            p_fake = 1 - tf.reduce_mean(tf.sigmoid(D_fake))

        g_results = {
            'g_loss': G_loss,
            'p(fake==0)': p_fake,
            'E[logit(fake)]': tf.reduce_mean(D_fake),
        }
        g_results = {cname + "_" + key: val for key, val in g_results.items()}

        res = dict()
        res["G_adversarial_loss"] = G_loss
        res["G_loss"] = G_loss
        res["G_p_fake"] = p_fake

        return res, g_results, dict()


"""
This loss does NOT work! It was an experiment to compute the loss on constraints
"""
class CanGeneratorLoss(_PolygonsFloorGeneratorLoss):

    def _forward(self, **graph_nodes):
        D_fake = graph_nodes["D_fake"]
        g_output_logit = graph_nodes["G_output_logit"]
        G_sample = graph_nodes["G_sample"]
        g_constraints = None
        normalization_factor = None

        log_w = tf.reshape(D_fake, [
                           self.experiment["NUM_BGAN_SAMPLES"], self.experiment["BATCH_SIZE"]])  # (20, 64)

        # notes.pdf, Appendix B, penultimate calculation
        log_g = -tf.reduce_sum((1. - G_sample) * pad_left(g_output_logit) +
                               pad_left(tf.nn.softplus(-g_output_logit)),
                               axis=[2, 3, 4])  # (20, 32)
        log_N = tf.log(tf.cast(log_w.shape[0], dtype=tf.float32))  # ()
        log_Z_est = tf.reduce_logsumexp(log_w - log_N, axis=0)  # (64,)
        log_w_tilde = log_w - pad_left(log_Z_est) - log_N  # (20, 64)
        w_tilde = tf.exp(log_w_tilde)  # (20, 64)

        log_resh_c = tf.reshape(g_constraints, [
                                self.experiment["NUM_BGAN_SAMPLES"], self.experiment["BATCH_SIZE"], -1])
        log_c = tf.reduce_mean(1 - log_resh_c, 2)  # (20,64)
        log_N_c = tf.log(tf.cast(log_c.shape[0], dtype=tf.float32))  # ()
        log_Z_c_est = tf.reduce_logsumexp(log_c - log_N_c, axis=0)  # (64,)
        log_c_tilde = log_c - pad_left(log_Z_c_est) - log_N_c  # (20, 64)
        c_tilde = tf.exp(log_c_tilde)  # (20, 64)

        G_loss_no_constraints = (1 - normalization_factor) * \
            -tf.reduce_mean(tf.reduce_sum(w_tilde * log_g, 0))
        G_loss_constraints = normalization_factor * \
            -tf.reduce_mean(tf.reduce_sum(c_tilde * log_g, 0))
        G_loss = G_loss_no_constraints + G_loss_constraints

        # TODO Find a way to compute loss on constraints
        with tf.name_scope("G_loss"):
            loss_summary = tf.summary.scalar("batch-by-batch", G_loss)
            loss_no_con = tf.summary.scalar(
                "loss_no_constr", G_loss_no_constraints)
            loss_con = tf.summary.scalar("loss_constr", G_loss_constraints)
            p_fake = 1 - tf.reduce_mean(tf.sigmoid(D_fake))
            p_fake_summary = tf.summary.scalar("p-fake-0", p_fake)

        g_results = {
            'g loss': G_loss,
            'p(fake==0)': p_fake,
            'E[logit(fake)]': tf.reduce_mean(D_fake),
        }

        res = dict()
        res["G_loss"] = G_loss
        res["G_loss_constraints"] = G_loss_constraints
        res["G_loss_no_constraints"] = G_loss_no_constraints

        return res, tf.summary.merge([loss_summary, loss_no_con, loss_con, p_fake_summary]), g_results, dict()
