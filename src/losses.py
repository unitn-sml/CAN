"""
Some losses that are not strictly related to a module/thesis/experiment.
"""
import tensorflow as tf
from base_layers import Loss
EPSILON = 1e-8

class RequiresRealFakeLoss(Loss):
    """
    A loss that requires D_real and D_fake to be in graph_nodes
    """

    def _pre_processing(self, **graph_nodes):
        assert "D_real" in graph_nodes
        assert "D_fake" in graph_nodes


class GanDiscriminatorLoss(RequiresRealFakeLoss):

    def _forward(self, **graph_nodes):
        D_real = graph_nodes["D_real"]
        D_fake = graph_nodes["D_fake"]

        D_loss = tf.contrib.gan.losses.wargs.modified_discriminator_loss(D_real, D_fake, label_smoothing=0.0,
                                                                         reduction=tf.losses.Reduction.MEAN)

        d_results = {
            'D_loss': D_loss,
            'E[logit(real)]': tf.reduce_mean(D_real),
            'p(real==1)': tf.reduce_mean(tf.sigmoid(D_real)),
            'p(fake==1)': tf.reduce_mean(tf.sigmoid(D_fake)),
        }

        res = dict()
        res["D_adversarial_loss"] = D_loss
        res["D_loss"] = D_loss
        res["D_p_real"] = d_results["p(real==1)"]

        cname = self.__class__.__name__
        d_results = {cname + "_" + key: val for key, val in d_results.items()}

        return res, d_results, dict()


class HingeDiscriminatorLoss(RequiresRealFakeLoss):

    def _forward(self, **graph_nodes):
        D_real = graph_nodes["D_real"]
        D_fake = graph_nodes["D_fake"]

        D_loss = tf.reduce_mean(tf.nn.relu(1. - D_real)) + tf.reduce_mean(tf.nn.relu(1. + D_fake))

        d_results = {
            'D_loss': D_loss,
            'E[logit(real)]': tf.reduce_mean(D_real),
            'p(real==1)': tf.reduce_mean(tf.sigmoid(D_real)),
            'p(fake==1)': tf.reduce_mean(tf.sigmoid(D_fake)),
        }

        res = dict()
        res["D_adversarial_loss"] = D_loss
        res["D_loss"] = D_loss
        res["D_p_real"] = d_results["p(real==1)"]

        cname = self.__class__.__name__
        d_results = {cname + "_" + key: val for key, val in d_results.items()}

        return res, d_results, dict()


class HingeGeneratorLoss(Loss):

    def _pre_processing(self, **graph_nodes):
        assert "D_fake" in graph_nodes

    def _forward(self, **graph_nodes):
        D_fake = graph_nodes["D_fake"]

        G_loss = -tf.reduce_mean(D_fake)

        g_results = {
            'g_loss': G_loss,
            'p(fake==0)': 1 - tf.reduce_mean(tf.sigmoid(D_fake)),
            'p(fake)': tf.reduce_mean(tf.sigmoid(D_fake)),
            'E[logit(fake)]': tf.reduce_mean(D_fake),
        }

        res = dict()
        res["G_adversarial_loss"] = G_loss
        res["G_loss"] = G_loss
        res["G_p_fake"] = g_results["p(fake==0)"]

        cname = self.__class__.__name__
        g_results = {cname + "_" + key: val for key, val in g_results.items()}

        return res, g_results, dict()


class GanGeneratorLoss(Loss):

    def _pre_processing(self, **graph_nodes):
        assert "G_sample" in graph_nodes
        assert "D_fake" in graph_nodes
        assert "G_output_logit" in graph_nodes

    def _forward(self, **graph_nodes):
        D_fake = graph_nodes["D_fake"]

        G_loss = tf.contrib.gan.losses.wargs.modified_generator_loss(D_fake, reduction=tf.losses.Reduction.MEAN)

        g_results = {
            'g_loss': G_loss,
            'p(fake==0)': 1 - tf.reduce_mean(tf.sigmoid(D_fake)),
            'p(fake)': tf.reduce_mean(tf.sigmoid(D_fake)),
            'E[logit(fake)]': tf.reduce_mean(D_fake),
        }

        res = dict()
        res["G_adversarial_loss"] = G_loss
        res["G_loss"] = G_loss
        res["G_p_fake"] = g_results["p(fake==0)"]

        cname = self.__class__.__name__
        g_results = {cname + "_" + key: val for key, val in g_results.items()}

        return res, g_results, dict()


class WassersteinDiscriminatorLoss(RequiresRealFakeLoss):

    def _forward(self, **graph_nodes):
        D_real = graph_nodes["D_real"]
        D_fake = graph_nodes["D_fake"]

        D_loss = tf.contrib.gan.losses.wargs.wasserstein_discriminator_loss(D_real, D_fake,
                                                                            reduction=tf.losses.Reduction.MEAN)

        d_results = {
            'D_loss': D_loss,
            'E[logit(real)]': tf.reduce_mean(D_real),
            'p(real==1)': tf.reduce_mean(tf.sigmoid(D_real)),
        }

        res = dict()
        res["D_adversarial_loss"] = D_loss
        res["D_loss"] = D_loss
        res["D_p_real"] = d_results["p(real==1)"]

        cname = self.__class__.__name__
        d_results = {cname + "_" + key: val for key, val in d_results.items()}

        return res, d_results, dict()


# class WassersteinGPDiscriminatorLoss(RequiresRealFakeLoss):
#
#     def _forward(self, **graph_nodes):
#         D_real = graph_nodes["D_real"]
#         D_fake = graph_nodes["D_fake"]
#
#         D_loss_pre_gp = tf.contrib.gan.losses.wargs.wasserstein_discriminator_loss(D_real, D_fake,
#                                                                                    reduction=tf.losses.Reduction.MEAN)
#
#         # Gradient Penalty
#         epsilon = tf.random_uniform(
#             shape=[self.experiment["BATCH_SIZE"], 1, 1, 1],
#             minval=0.,
#             maxval=1.)
#         X_hat = graph_nodes["X"] + epsilon * (graph_nodes["G_sample"] - graph_nodes["X"])
#
#         gsample = graph_nodes["G_sample"]
#         graph_nodes["G_sample"] = X_hat
#
#         D_X_hat, _ = graph_nodes["discriminator"](**graph_nodes)
#         D_X_hat = D_X_hat["D_fake"]
#
#         grad_D_X_hat = tf.gradients(D_X_hat, [X_hat])[0]
#         red_idx = list(range(1, X_hat.shape.ndims))
#         slopes = tf.sqrt(tf.reduce_sum(tf.square(grad_D_X_hat), reduction_indices=red_idx))
#         gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
#         D_loss = D_loss_pre_gp + 10 * gradient_penalty
#
#         graph_nodes["G_sample"] = gsample
#
#         cname = self.__class__.__name__
#         with tf.name_scope(cname):
#             p_real = tf.reduce_mean(tf.sigmoid(D_real))
#
#         d_results = {
#             'D_loss': D_loss_pre_gp,
#             'p(real==1)': p_real,
#             'p(fake==1)': tf.reduce_mean(tf.sigmoid(D_fake)),
#             'E[logit(real)]': tf.reduce_mean(D_real),
#         }
#         d_results = {cname + "_" + key: val for key, val in d_results.items()}
#
#         res = dict()
#         res["D_adversarial_loss"] = D_loss_pre_gp
#         res["D_loss"] = D_loss
#         res["D_p_real"] = p_real
#
#         return res, d_results


class WassersteinGPDiscriminatorLoss(Loss):

    def _pre_processing(self, **graph_nodes):
        assert "D_real" in graph_nodes
        assert "D_fake" in graph_nodes
        assert "X" in graph_nodes
        assert "G_sample" in graph_nodes
        assert "z" in graph_nodes
        assert "discriminator" in graph_nodes

    def _forward(self, **graph_nodes):
        D_real = graph_nodes["D_real"]
        D_fake = graph_nodes["D_fake"]
        data = graph_nodes["X"]
        gdata = graph_nodes["G_sample"]
        z = graph_nodes["z"]
        disc = graph_nodes["discriminator"]

        def discriminator_fn(inputs, gen_inputs):
            inputs_dict = {"G_sample": inputs}
            outs, _ = disc(**inputs_dict)
            return outs["D_fake"]

        gradient_penalty = tf.contrib.gan.losses.wargs.wasserstein_gradient_penalty(
            real_data=data,
            generated_data=gdata,
            generator_inputs=z,
            discriminator_fn=discriminator_fn,
            discriminator_scope="can",
            reduction=tf.losses.Reduction.MEAN
        )

        D_loss_pre_gp = tf.reduce_mean(D_fake) - tf.reduce_mean(D_real)
        D_loss = D_loss_pre_gp + 10 * gradient_penalty

        d_results = {
            'D_loss': D_loss,
            'E[logit(real)]': tf.reduce_mean(D_real),
            'p(real==1)': tf.reduce_mean(tf.sigmoid(D_real)),
        }

        res = dict()
        res["D_adversarial_loss"] = D_loss_pre_gp
        res["D_loss"] = D_loss
        res["D_p_real"] = d_results["p(real==1)"]

        cname = self.__class__.__name__
        d_results = {cname + "_" + key: val for key, val in d_results.items()}

        return res, d_results, dict()


class WassersteinGeneratorLoss(Loss):

    def _pre_processing(self, **graph_nodes):
        assert "G_sample" in graph_nodes
        assert "D_fake" in graph_nodes
        assert "G_output_logit" in graph_nodes

    def _forward(self, **graph_nodes):
        D_fake = graph_nodes["D_fake"]
        D_real = graph_nodes["D_real"]

        G_loss = tf.contrib.gan.losses.wargs.wasserstein_generator_loss(D_fake, reduction=tf.losses.Reduction.MEAN)

        g_results = {
            'g_loss': G_loss,
            'p(fake==0)': 1 - tf.reduce_mean(tf.sigmoid(D_fake)),
            'p(fake)': tf.reduce_mean(tf.sigmoid(D_fake)),
            'E[logit(fake)]': tf.reduce_mean(D_fake),
        }

        res = dict()
        res["G_adversarial_loss"] = G_loss
        res["G_loss"] = G_loss
        res["G_p_fake"] = g_results["p(fake==0)"]

        cname = self.__class__.__name__
        g_results = {cname + "_" + key: val for key, val in g_results.items()}

        return res, g_results, dict()


class JSBatchLoss(Loss):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_height, self.sample_width, self.sample_channels = self.experiment["SHAPE"]

    def _pre_processing(self, **graph_nodes):
        assert "G_output_logit" in graph_nodes, "Expected to find G_output_logit in graph nodes, which is the raw " \
                                                "output of the Generator (This loss will take care of applying the " \
                                                "softmax function)"

    def _forward(self, **graph_nodes):
        """ Need some magic to iterate over first undefined dimension. """

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

        avg_distrib = (probs_a + probs_b) / 2

        kl_div_1 = tf.reduce_sum(tf.math.multiply(probs_a, tf.math.log(tf.math.divide(probs_a, avg_distrib))), axis=-1)
        kl_div_2 = tf.reduce_sum(tf.math.multiply(probs_b, tf.math.log(tf.math.divide(probs_b, avg_distrib))), axis=-1)

        JS_distance = tf.math.sqrt(0.5 * kl_div_1 + 0.5 * kl_div_2)

        res = tf.maximum(EPSILON, -tf.reduce_sum(JS_distance) / 2)
        # res = tf.constant(0.0, dtype=tf.float32)

        nodes = { "G_loss": res, "JS_loss": res }

        nodes_to_log = { "debug_JS_loss": res, 
            "debug_probs": tf.reduce_sum(probs),
            "debug_probs_a": tf.reduce_sum(probs_a),
            "debug_probs_a": tf.reduce_sum(probs_b),
            "debug_avg_distrib": tf.reduce_sum(avg_distrib),
            "debug_kl_div_1": tf.reduce_sum(kl_div_1),
            "debug_kl_div_2": tf.reduce_sum(kl_div_2),
            "debug_JS_distance": tf.reduce_sum(JS_distance)
        }

        return nodes, nodes_to_log, dict()


class L1NormLoss(Loss):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_height, self.sample_width, self.sample_channels = self.experiment["SHAPE"]

    def _pre_processing(self, **graph_nodes):
        assert "G_output_logit" in graph_nodes, "Expected to find G_output_logit in graph nodes, which is the raw " \
                                                "output of the Generator (This loss will take care of applying the " \
                                                "softmax function)"

    def _forward(self, **graph_nodes):
        """ Need some magic to iterate over first undefined dimension.
        We will broadcast each batch on two new dimensions in a way like
        [bs, -1] -> 

        a: [1, bs, -1]
        b: [bs, 1, -1]

        |a-b|: [bs, bs, -1] each element paired (2 times) with all the other elements.
        Finally divide by 2 to have your result.
        """

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

        nodes = { "G_loss": -l1_norm, "l1_norm": l1_norm }

        nodes_to_log = { "l1_norm": l1_norm }

        return nodes, nodes_to_log, dict()