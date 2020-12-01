"""
This files takes an already trained models and extract N samples.
"""


from time import time
from datetime import timedelta
from os import makedirs, walk, sep as path_separator
import numpy as np
import tensorflow as tf
from tensorflow.python.framework.errors_impl import (NotFoundError as CheckpointNotFoundError)
from tensorflow.python import debug as tf_debug
from trainer import Trainer

"""
Signature of methods like train, validate, test, etc. have been kept as generic as possible, meaning
that there is stuff that is currently unused so that if a method is overridden it can be modified and be made
to use all those arguments without modifying the caller.
"""
from utils.utils_common import first_arg_null_safe, update_dict_of_lists, safe_merge, make_read_only, \
    save_random_states, load_random_states, remove_folder, compress_folder, set_seeds
from utils.custom_loggers import TrainerLogger


class Evaluator(Trainer):
    def __init__(self, experiment, tf_session_config, complete_logging, number_of_samples):
        self.experiment = experiment

        self.logger = TrainerLogger(self.experiment, complete_logging)

        self.logger.log_version()
        self.logger.log_random_states("Starting evaluation")
        self.number_of_samples = number_of_samples

        """
        to generate some evaluation samples, the generator is more than enough.
        """
        with tf.variable_scope("can"):
            # leaves that might be used by functions such as generator, discriminator, losses, etc.
            graph_nodes = dict()

            # another dict to store nodes that needs to be initialized after tf.global_variable_initializer
            init_nodes = dict()

            ######################## GENERATOR ##########################################
            generator = experiment["GENERATOR"](experiment)
            G_graph_nodes, G_to_log, G_init_nodes = generator(**graph_nodes)
            graph_nodes = safe_merge(graph_nodes, G_graph_nodes)
            init_nodes = safe_merge(init_nodes, G_init_nodes)
        self.computables = [
            Computable(experiment, None, None, None, graph_nodes=graph_nodes) for
            Computable in experiment["COMPUTABLES"]]
        self.graph_nodes = make_read_only(graph_nodes)

        self._init_saver()

        with tf.Session(config=tf_session_config) as session:
            self.session = session
            self.session.run(tf.global_variables_initializer())

            # run all custom initialization functions for all nodes in init_nodes
            for (key, node) in init_nodes.items():
                self.logger.info("Initializing weights of node: %s" % key)
                node._init_weights(**graph_nodes)

            # load preexisting session! it must exists, we are in evaluation mode
            self._load_session()

            # it is easier to keep track of progress if evaluation is done on a fixed noise vector
            np.random.seed(seed=12321)

            self.eval_z = self._generate_evaluation_z_vector(self.experiment["BATCH_SIZE"])
            self.logger.log_seed_custom(self.eval_z, what="Evaluation noise")

            # make the graph read-only to prevent memory leaks
            tf.get_default_graph().finalize()
            self.logger.log_random_states("After graph finalization")
            self.logger.log_trainable_variables("After graph finalization")
            self.logger.log_trainable_variables_weights("After graph finalization")

            self.logger.info("Generating evaluation samples")
            self._generate_evaluation_images()

    def _load_session(self):
        """
        if there is a checkpoint that can be restored, restore it and restart training from that checkpoint
        """
        try:
            # list available checkpoints and pick the one from the latest epoch
            _, dirs, _ = next(walk(self.experiment["CHECKPOINT_FOLDER"]))
            dirs.sort()
            last_epoch = self.experiment["CHECKPOINT_FOLDER"] + dirs[-1] + "/"
            checkpoint_model = last_epoch + self.experiment["CHECKPOINT_FILE"]
            self.saver.restore(self.session, checkpoint_model)
            load_random_states(self.experiment, last_epoch)

            msg = "Successfully loaded checkpoint file: {}\n"
            self.logger.info(msg.format(checkpoint_model))
        except (CheckpointNotFoundError, IndexError) as ex:
            if type(ex) == IndexError:
                self.logger.info("Unable to find any checkpoint")
            else:
                self.logger.exception("Unable to load checkpoint correctly")
            # raise error if pretrined model is not found
            raise AssertionError("In evaluation mode a pretrained model MUST be present.")

    def _generate_evaluation_images(self):
        if self.experiment.plot_data is not None:
            """
            given the starting random evaluation vector z, generate a new bunch of images that
            will be saved to the respective folder
            """
            self.logger.log_random_states("Evaluation images generation")

            # save images to tensorboard starting from constant eval_z noise
            i = 0
            j = 0
            samples = []
            while i < self.number_of_samples + self.experiment["BATCH_SIZE"]:
                feed_dict = {
                    # generate batch_size random input noises
                    self.graph_nodes["z"]: self.eval_z
                }
                for computable in self.computables:
                    computable.compute(feed_dict, None)
                G_sample = self.session.run(self.graph_nodes["G_sample"], feed_dict=feed_dict)

                """
                We want a batch of images both if an architecture G_sample corresponds exactly to a batch of images
                or if its actually of shape [batch, samples, shape] because the architecture samples a certain number
                of elements for each element of a batch (like bgan).
                In the second case, we would like to get take a single discretized sample from each batch element, as
                if we produced an output of [batch, 1, shape].
                So first we reshape, then we select the right indexes.
                """
                G_sample_batch = np.reshape(G_sample, ([-1] + self.experiment["SHAPE"]))
                self.logger.log_seed_custom(G_sample_batch, "Evaluation images generation", "images")
                if self.experiment["DATASET_TYPE"] == "random_formula":
                    samples.extend(G_sample_batch)
                else:
                    self.experiment.plot_data(G_sample_batch, j, evaluation=True)
                msg = "Generated samples saved in folder: {}"
                self.logger.info(msg.format(self.experiment["OUTPUT_FOLDER"]))
                j +=1
                i += self.experiment["BATCH_SIZE"]
            if self.experiment["DATASET_TYPE"] == "random_formula":
                self.experiment.plot_data(np.array(samples[:self.number_of_samples]), "test")
        else:
            self.logger.warning("We are in evaluation mode and not function (plot_eval_data) is defined to save generated samples")


    def _generate_evaluation_z_vector(self, number):
        """
        number: the number of samples that has to be generated
        """
        eval_z = self._generate_input_z_vector(number)
        return eval_z
