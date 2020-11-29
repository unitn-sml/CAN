import numpy as np
import random
import xxhash
import tensorflow as tf

from datetime import timedelta
from utils.utils_common import min_max


class TrainerLogger:
    """
    Class to encapsulate/gather all trainer level logging (note that this logging
    is not related to the logging of computables, which is done personally by each computable class instance).
    It's API is either about feeding the instance class data or asking if something should be logged, i.e.
    if tf_summaries should be run with the ops.
    """

    def __init__(self, experiment, complete_logging=False):
        self.logger = experiment["LOGGER"]
        self.complete_logging = complete_logging
        self.LOG_TRAINING_EVERY = 1  # batches number
        self.LOG_SUMMARIES_EVERY = 20  # batches number
        self.PLOT_SAMPLES_EVERY = 100  # epochs number
        self.STORE_CHECKPOINT_EVERY = 1000  # epochs number
        self.TEST_EVERY = 20  # epochs number
        self.VALIDATE_EVERY = 999  # epochs number

        self.seed_logger = experiment["SEED_LOGGER"]
        msg = "Please remember that hashing might have different inputs having " \
              "the same output. This logging file contains, among other things, " \
              "hashes, used for readability in place of the whole input, take them with a grain of salt."
        self.seed_logger.info(msg)
        if not complete_logging:
            msg = "Complete logging is not enabled (run with --complete_logging to enable), seed logging will " \
                  "not continue during training."
            self.seed_logger.info(msg)

    def info(self, msg):
        """
        Logs the msg with "info" level.
        :param msg:
        """
        self.logger.info(msg)

    def exception(self, msg):
        """
        Logs the msg with "exception" level.
        :param msg:
        """
        self.logger.exception(msg)

    def debug(self, msg):
        """
        Logs the msg with "debug" level.
        :param msg:
        """
        self.logger.debug(msg)

    def log_loss(self, fn_name, step_type, epoch, losses, norm_factor):
        """
        Log a loss, computing and logging it's average in the batch, min, max.

        :param fn_name: Name of the loss (G_loss, D_loss, not enforced).
        :param step_type: Type of step, (training, validation, testing, not enforced).
        :param epoch: Epoch number.
        :param losses: Actual tensor with results.
        :param norm_factor: Normalization factor to apply to the sum of losses (needed given that losses might be computed
        over many GEN or DISC iterations).
        """
        mb_num = len(losses)
        loss_mb_min, loss_mb_max = min_max(losses)
        loss_mb_min /= norm_factor
        loss_mb_max /= norm_factor
        avg_loss = sum(losses) / (mb_num * norm_factor)
        msg = "{} epoch {}, {} on batches: avg {}; min {}; max {}."
        self.logger.info(msg.format(step_type.upper(), epoch, fn_name, avg_loss, loss_mb_min, loss_mb_max))

    def log_epoch_info(self, step_type, epoch, results, norm_factor):
        """
        Log epoch info.

        :param step_type: Type of step, (training, validation, testing, not enforced).
        :param epoch: Epoch number.
        :param results: Results to lg.
        :param norm_factor: Normalization factor to apply to the results (needed given that results might be computed
        over many GEN or DISC iterations).
        """
        sorted_keys = sorted(results, key=lambda x: x[0])
        result_summary = dict((k, np.sum(results[k]) / (len(results[k]) * norm_factor)) for k in sorted_keys)
        msg = "{} epoch results: {}: {}"
        self.logger.info(msg.format(step_type.upper(), epoch, result_summary))

    def log_epoch_progress(self, step_type, epoch, start_idx, bs, total, mbs_duration):
        """
        Log epoch progress.

        :param step_type: Type of step, (training, validation, testing, not enforced).
        :param epoch: Epoch number.
        :param start_idx: Starting index of the current batch of data.
        :param bs: Batch size.
        :param total: Total number of samples that are examined in an epoch.
        :param mbs_duration: How long the batch took to compute.
        """
        processed_examples = start_idx + min(bs, total - start_idx)
        msg = "{} step epoch {}, processed {}/{} samples: {:.2f}%. Epoch ETA: {}"
        if self.do_log_mb_progress(processed_examples, bs, total):
            missing_mbs = (total - processed_examples) // bs
            mb_mean_time = np.mean(mbs_duration)
            epoch_ETA = timedelta(seconds=mb_mean_time * missing_mbs)
            self.logger.info(
                msg.format(step_type.upper(), epoch, processed_examples, total, processed_examples * 100 / total,
                           str(epoch_ETA)))

    """
    Methods to encapsulate logging related to seeds, random states, indices, etc. To help in debugging.
    Logging can be costly, especially when done on the weights of the neural network, which need to be evaluated.
    """

    def log_version(self):
        self.seed_logger.info("xxhash VERSION %s" % xxhash.VERSION)
        self.seed_logger.info("xxhash XXHASH_VERSION %s" % xxhash.XXHASH_VERSION)

    def log_dataset(self, training_data, test_data, validation_data):
        self.seed_logger.info("Train/test/validation data lengths: %s, %s, %s" % (
            len(training_data), len(test_data), len(validation_data)))
        self.log_seed_custom(training_data, what="Train set")
        self.log_seed_custom(test_data, what="Test set")
        self.log_seed_custom(validation_data, what="Validation set")

    def log_random_states(self, when=""):
        if not self.complete_logging:
            return
        self.log_seed_custom(random.getstate(), when, "Random state")
        self.log_seed_custom(np.random.get_state(), when, "Numpy Random state")

    def log_trainable_variables(self, when=""):
        if not self.complete_logging:
            return
        trainable_variables = [v for v in tf.trainable_variables()]
        self.log_seed_custom(trainable_variables, when, "trainable variables")

    def log_trainable_variables_weights(self, when=""):
        if not self.complete_logging:
            return
        trainable_variables_weights = [v.eval() for v in tf.trainable_variables()]
        self.log_seed_custom(trainable_variables_weights, when, "trainable variables weights")

    def log_seed_custom(self, data, when="", what=""):
        """
        Log whatever, data must have the __str__ method.
        Logging is made in the form of when+what+"hash <hash of data"
        If there is no space as the last character of when and/or what a space will be appended.
        :param data:
        :param when:
        :param what:
        :return:
        """
        if not self.complete_logging:
            return
        when = when if (len(when) == 0 or when[-1] == " ") else when + " "
        what = what if (len(what) == 0 or what[-1] == " ") else what + " "
        self.seed_logger.info("%s%shash %s" % (when, what, xxhash.xxh64(str(data)).intdigest()))

    """
    Methods to either help this class instance decide if something should be logged or to help the trainer
    decide if some operations should be run, for example some kind of summaries.
    """

    def do_log_mb_progress(self, processed, bs, examples):
        # when statistics should be printed during training
        return (processed % (bs * self.LOG_TRAINING_EVERY)) == 0 or processed + bs >= examples

    def do_log_summaries(self, processed, bs, examples):
        # when tensorboard summaries should be collected during training
        return (processed % (bs * self.LOG_SUMMARIES_EVERY)) == 0 \
               or processed + bs >= examples

    def do_plot_samples(self, epoch):
        # when evaluation noise should be used to create images during training
        return epoch % self.PLOT_SAMPLES_EVERY == 0

    def do_store_checkpoint(self, epoch):
        # when checkpoints should be saved during training
        return epoch % self.STORE_CHECKPOINT_EVERY == 0

    def do_test_network(self, epoch):
        # when ANN should be evaluated on test data
        return epoch % self.TEST_EVERY == 0

    def do_validate_network(self, epoch):
        # when ANN should be evaluated on validation data
        return epoch % self.VALIDATE_EVERY == 0
