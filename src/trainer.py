from time import time
from datetime import timedelta
from os import makedirs, walk, sep as path_separator
import numpy as np
import math
import tensorflow as tf
from tensorflow.python.framework.errors_impl import (NotFoundError as CheckpointNotFoundError)
from tensorflow.python import debug as tf_debug

"""
Signature of methods like train, validate, test, etc. have been kept as generic as possible, meaning
that there is stuff that is currently unused so that if a method is overridden it can be modified and be made
to use all those arguments without modifying the caller.
"""
from utils.utils_common import first_arg_null_safe, update_dict_of_lists, safe_merge, make_read_only, \
    save_random_states, load_random_states, remove_folder, compress_folder, set_seeds
from utils.custom_loggers import TrainerLogger


class Trainer:
    def __init__(self, experiment, tf_session_config, complete_logging):
        self.experiment = experiment

        self.logger = TrainerLogger(self.experiment, complete_logging)

        self.training_data = experiment.training_data
        self.validation_data = experiment.validation_data
        self.test_data = experiment.test_data

        self.logger.log_version()
        self.logger.log_dataset(self.training_data, self.test_data, self.validation_data)
        self.logger.log_random_states("Start of training")

        # list of indices of the whole training dataset
        self.training_batches_indices = np.arange(len(self.training_data))

        # keep track of the current epoch, updated when training
        self.curr_epoch_var = tf.Variable(0, trainable=False, dtype=tf.int32)
        self.increment_curr_epoch_op = tf.assign_add(self.curr_epoch_var, 1)

        with tf.variable_scope("can"):
            # leaves that might be used by functions such as generator, discriminator, losses, etc.
            graph_nodes = dict()
            graph_nodes["current_epoch"] = self.curr_epoch_var
            graph_nodes["G_global_step"] = tf.Variable(0, name='G_global_step', trainable=False)
            graph_nodes["D_global_step"] = tf.Variable(0, name='D_global_step', trainable=False)

            # another dict to store nodes that needs to be initialized after tf.global_variable_initializer
            init_nodes = dict()

            ######################## GENERATOR ##########################################
            generator = experiment["GENERATOR"](experiment)
            G_graph_nodes, G_to_log, G_init_nodes = generator(**graph_nodes)
            graph_nodes = safe_merge(graph_nodes, G_graph_nodes)
            init_nodes = safe_merge(init_nodes, G_init_nodes)

            ######################## DISCRIMINATOR ######################################
            discriminator = experiment["DISCRIMINATOR"](experiment)
            D_graph_nodes, D_to_log, D_init_nodes = discriminator(**graph_nodes)
            graph_nodes = safe_merge(graph_nodes, D_graph_nodes)
            init_nodes = safe_merge(init_nodes, D_init_nodes)

            ######################## GENERATOR LOSS #####################################
            G_loss = None
            # unpack each (loss, weight pair) in the list of losses
            for generator_loss, weight in experiment["GENERATOR_LOSS"]:
                generator_loss = generator_loss(experiment, weight=weight)
                G_loss_nodes, G_loss_to_log, G_loss_init_nodes = generator_loss(**graph_nodes)

                assert "G_loss" in G_loss_nodes, "Expected a node named 'G_loss' in the output nodes of %s" \
                                                 % generator_loss.__class__.__name__

                """
                Take out 'G_loss' out of the graph_nodes so that the rest of the nodes can merge and we won't
                have merging problems due to having 'G_loss' in many losses.
                The node is rescaled by its weight, and summed to the total G_loss.
                """
                G_loss_current_node = G_loss_nodes.pop("G_loss") * tf.constant(weight, dtype=tf.float32)
                G_loss = G_loss_current_node if G_loss is None else (G_loss + G_loss_current_node)

                # merge all the nodes but G_loss
                graph_nodes = safe_merge(graph_nodes, G_loss_nodes)

                # merge all the init nodes
                init_nodes = safe_merge(init_nodes, G_loss_init_nodes)

                # we expect stuff to be named properly so no clashes happen
                G_to_log = safe_merge(G_to_log, G_loss_to_log)
            # add the sum of losses as the loss node
            graph_nodes["G_loss"] = G_loss

            ######################## DISCRIMINATOR LOSS #################################
            D_loss = None
            # unpack each (loss, weight pair) in the list of losses
            for discriminator_loss, weight in experiment["DISCRIMINATOR_LOSS"]:
                discriminator_loss = discriminator_loss(experiment, weight=weight)
                D_loss_nodes, D_loss_to_log, D_loss_init_nodes = discriminator_loss(**graph_nodes)

                assert "D_loss" in D_loss_nodes, "Expected a nome named 'D_loss' in the output nodes of %s" \
                                                 % discriminator_loss.__class__.__name__

                """
                Take out 'D_loss' out of the graph_nodes so that the rest of the nodes can merge and we won't
                have merging problems due to having 'D_loss' in many losses.
                The node is rescaled by its weight, and summed to the total D_loss.
                """
                D_loss_current_node = D_loss_nodes.pop("D_loss") * tf.constant(weight, dtype=tf.float32)
                D_loss = D_loss_current_node if D_loss is None else (D_loss + D_loss_current_node)

                # merge all the nodes but D_loss
                graph_nodes = safe_merge(graph_nodes, D_loss_nodes)

                # merge all the init nodes
                init_nodes = safe_merge(init_nodes, D_loss_init_nodes)

                # we expect stuff to be named properly so no clashes happen
                D_to_log = safe_merge(D_to_log, D_loss_to_log)
            # add the sum of losses as the loss node
            graph_nodes["D_loss"] = D_loss

            ################################ SOLVERS ####################################
            # get G/D parameters and optimize them
            theta_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "can/generator")
            theta_D = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "can/discriminator")
            generator_optimizer = experiment["GENERATOR_SOLVER"](experiment)
            discriminator_optimizer = experiment["DISCRIMINATOR_SOLVER"](experiment)
            G_solver = generator_optimizer(theta_G, **graph_nodes)
            D_solver = discriminator_optimizer(theta_D, **graph_nodes)

            ############################## COMPUTABLES ##################################
            self.computables = [
                Computable(experiment, self.training_data, self.validation_data, self.test_data, graph_nodes) for
                Computable in experiment["COMPUTABLES"]]

            ######################### GENERATOR STATISTICS ##############################
            # merge each statistic on the generator in the G_summaries collection
            G_summaries = None
            for statistic in experiment["GENERATOR_STATISTICS"]:
                statistic = statistic(experiment)
                summaries = statistic(**graph_nodes)

                if G_summaries is not None:
                    if summaries is not None:
                        G_summaries = tf.summary.merge([G_summaries, summaries])
                else:
                    G_summaries = summaries

            ######################## DISCRIMINATOR STATISTICS ###########################
            # merge each statistic on the discriminator in the D_summaries collection
            D_summaries = None
            for statistic in experiment["DISCRIMINATOR_STATISTICS"]:
                statistic = statistic(experiment)
                summaries = statistic(**graph_nodes)

                if D_summaries is not None:
                    if summaries is not None:
                        D_summaries = tf.summary.merge([D_summaries, summaries])
                else:
                    D_summaries = summaries

        self.graph_nodes = make_read_only(graph_nodes)
        D_loss = graph_nodes["D_loss"]
        G_loss = graph_nodes["G_loss"]
        self._init_saver()

        with tf.Session(config=tf_session_config) as session:
            self.session = session
            if self.experiment["DEBUG"]:
                self.session = tf_debug.TensorBoardDebugWrapperSession(self.session, 'localhost:6007')
            self.session.run(tf.global_variables_initializer())

            # run all custom initialization functions for all nodes in init_nodes
            for (key, node) in init_nodes.items():
                self.logger.info("Initializing weights of node: %s" % key)
                node._init_weights(**graph_nodes)

            # load preexisting session if possible
            self._maybe_load_session()
            # define where to collect summaries
            self.tb_writer = tf.summary.FileWriter(experiment["TB_EXPERIMENT_FOLDER"], self.session.graph)
            # make tb_writer None-safe for convenience (we log summaries only
            # periodically during training: this tb_writer will properly
            # handle the absence of summaries)
            self.tb_writer.add_summary = first_arg_null_safe(self.tb_writer.add_summary)

            # it is easier to keep track of progress if evaluation is done on a fixed noise vector
            self.eval_z = self._generate_evaluation_z_vector(experiment)
            self.logger.log_seed_custom(self.eval_z, what="Evaluation noise")

            self.curr_epoch = self.curr_epoch_var.eval(self.session)
            self.logger.info("Starting from epoch: {}".format(self.curr_epoch))

            # make the graph read-only to prevent memory leaks
            tf.get_default_graph().finalize()
            self.logger.log_random_states("After graph finalization")
            self.logger.log_trainable_variables("After graph finalization")
            self.logger.log_trainable_variables_weights("After graph finalization")

            if self.curr_epoch != experiment["LEARNING_EPOCHS"]:
                self._train(G_loss, G_solver, G_summaries, G_to_log, D_loss, D_solver, D_summaries, D_to_log)
                self.logger.info("Training completed.")

            self.logger.info("Performing final test with fixed seed...")
            self._test(G_loss, G_solver, G_summaries, G_to_log, D_loss, D_solver,
                       D_summaries, D_to_log, self.experiment["ANN_SEED"])

    def _init_saver(self):
        """
        initialize the saver to allow the save/restore of a run
        """
        # allow the training session to be saved and restored
        self.saver = tf.train.Saver(max_to_keep=None)

    def _store_session(self):
        """
        store the actual training session, allowing to restore it later.
        """
        if self.curr_epoch == 0:  # happens if you add constraints from epoch 0
            return  # no training has occurred: no need to store anything
        epoch_number = str(self.curr_epoch - 1).zfill(6)
        epoch_folder = self.experiment["CHECKPOINT_FOLDER"] + "epoch_" + epoch_number + path_separator
        makedirs(epoch_folder, exist_ok=True)

        # save the session
        checkpoint_path = epoch_folder + self.experiment["CHECKPOINT_FILE"]
        save_path = self.saver.save(self.session, checkpoint_path)

        # save the experiment
        save_random_states(self.experiment, epoch_folder)

        self.logger.info("Model saved in file: " + save_path)

    def _maybe_load_session(self):
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
            msg = "Starting from scratch. Unable to load checkpoint from: {}"
            self.logger.info(msg.format(self.experiment["CHECKPOINT_FOLDER"]))
            # tensorboard experiment folder is cleaned iff there is no
            # checkpoint, because it is desirable to keep events from previous
            # runs to display more informative statistics
            remove_folder(self.experiment["TB_EXPERIMENT_FOLDER"])
            makedirs(self.experiment["TB_EXPERIMENT_FOLDER"], exist_ok=True)

    def _discriminator_step(self, ops, real_data, real_tensors_indices, step_type):
        """
        Perform a step of the discriminator, first by building the feed dict with noise and the passed real data, mapping
        graph_nodes["z"] to the noise and graph_nodes["X"] to the real data, then calling all the computables in order.
        After the feed dictionary is completely the passed ops are run.
        Generator and Discriminator step are identical, but are kept separated to allow for easier customization
        and evolution of the code.
        Summaries from the ops are automatically added to tb and thus not returned.

        :param ops: Operations to run. Expected to be a list of length 4, with the third element being tb summaries.
        :param real_data: Batch of real data. Usage depends on your computational graph.
        :param real_tensors_indices: Indices of the passed real data.
        :param step_type: In what type of step are we, (training, evaluation, testing).
        :return: D loss and results (first and last ops in the list), as defined by the operations to run.
        """
        self.logger.log_random_states("Discriminator step")

        feed_dict = dict()
        feed_dict[self.graph_nodes["z"]] = self._generate_input_z_vector(real_data.shape[0])
        feed_dict[self.graph_nodes["X"]] = real_data
        # dict that computables can use to communicate between each other and/or with the trainer script
        shared_dict = dict()

        # add more values to the feed dict using Computables, if there are any
        for computable in self.computables:
            computable.compute(feed_dict, shared_dict, self.curr_epoch, real_tensors_indices, False, step_type)

        for k in sorted(feed_dict, key=lambda x: str(x)):
            self.logger.log_seed_custom(feed_dict[k], "Discriminator step", str(k))

        # run optimizer
        D_loss_curr, _, summaries, D_results = self.session.run(ops, feed_dict)
        self.tb_writer.add_summary(summaries, self.curr_epoch)
        return D_loss_curr, D_results

    def _generator_step(self, ops, real_data, real_tensors_indices, step_type):
        """
        Perform a step of the generator, first by building the feed dict with noise and the passed real data, mapping
        graph_nodes["z"] to the noise and graph_nodes["X"] to the real data, then calling all the computables in order.
        After the feed dictionary is completely the passed ops are run.
        Generator and Discriminator step are identical, but are kept separated to allow for easier customization
        and evolution of the code.
        Summaries from the ops are automatically added to tb and thus not returned.

        :param ops: Operations to run. Expected to be a list of length 4, with the third element being tb summaries.
        :param real_data: Batch of real data. Usage depends on your computational graph.
        :param real_tensors_indices: Indices of the passed real data.
        :param step_type: In what type of step are we, (training, evaluation, testing).
        :return: G loss and results (first and last ops in the list), as defined by the operations to run.
        """
        self.logger.log_random_states("Generator step")

        feed_dict = dict()
        feed_dict[self.graph_nodes["z"]] = self._generate_input_z_vector(real_data.shape[0])
        feed_dict[self.graph_nodes["X"]] = real_data
        # dict that computables can use to communicate between each other and/or with the trainer script
        shared_dict = dict()

        # add more values to the feed dict using Computables, if there are any
        for computable in self.computables:
            computable.compute(feed_dict, shared_dict, self.curr_epoch, real_tensors_indices, True, step_type)

        for k in sorted(feed_dict, key=lambda x: str(x)):
            self.logger.log_seed_custom(feed_dict[k], "Generator step", str(k))

        # run optimizer
        G_loss_curr, _, summaries, G_results = self.session.run(ops, feed_dict)
        self.tb_writer.add_summary(summaries, self.curr_epoch)
        return G_loss_curr, G_results

    def _generate_evaluation_images(self):
        if self.experiment.plot_data is not None and self.experiment["EVAL_SAMPLES"] > 0:
            """
            given the starting random evaluation vector z, generate a new bunch of images that
            will be saved to the respective folder
            """
            self.logger.log_random_states("Evaluation images generation")

            # save images to tensorboard starting from constant eval_z noise
            feed_dict = {
                # generate batch_size random input noises
                self.graph_nodes["z"]: self.eval_z
            }

            # dict that computables can use to communicate between each other and/or with the trainer script
            shared_dict = dict()
            for computable in self.computables:
                if computable.needed_for_evaluation_images:
                    computable.compute(feed_dict, shared_dict, self.curr_epoch, None, True, "evaluation")

            # if some computable has already computed G_sample
            if self.graph_nodes["G_sample"] in shared_dict:
                G_sample = shared_dict[self.graph_nodes["G_sample"]]
            else:
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
            nsamples = G_sample_batch.shape[0]
            indexes = list(range(0, nsamples, nsamples // self.experiment["EVAL_SAMPLES"]))
            G_sample_batch = G_sample_batch[indexes]
            self.logger.log_seed_custom(G_sample_batch, "Evaluation images generation", "images")

            self.experiment.plot_data(G_sample_batch, self.curr_epoch + 1)
            msg = "Generated samples saved in folder: {}"
            self.logger.info(msg.format(self.experiment["OUTPUT_FOLDER"]))

    def _train(self, G_loss_op, G_solver_op, G_summaries_op, G_results_op, D_loss_op, D_solver_op, D_summaries_op,
               D_results_op):
        """
        the main train function. for each epoch:
        1) check if constraints on fake data has to be computed
        2) shuffle the dataset
        3) run a train epoch giving as operations the losses and the solvers
        4) update the epoch counter
        5) log training infos
        6) if needed, do validation, testing or save the session
        finally, save the last session and compress results before saving them to the disk

        """
        # used to predict training epoch ETA
        epochs_duration = []

        # ANN training
        while self.curr_epoch < self.experiment["LEARNING_EPOCHS"]:
            epoch_start_time = time()

            self.logger.log_random_states("Start of train epoch")
            self.logger.log_trainable_variables("Start of train epoch")
            self.logger.log_trainable_variables_weights("Start of train epoch")
            np.random.shuffle(self.training_batches_indices)
            self.logger.log_seed_custom(self.training_batches_indices, "Start of train epoch", "batch indices")

            self.logger.info("Starting TRAINING epoch {}".format(self.curr_epoch))
            if self.experiment["WGAN"]:
                D_losses, G_losses, epoch_results = self._run_train_epoch_wgan(G_loss_op, G_solver_op, G_summaries_op,
                                                                      G_results_op, D_loss_op, D_solver_op,
                                                                      D_summaries_op, D_results_op)
            else:
                D_losses, G_losses, epoch_results = self._run_train_epoch(G_loss_op, G_solver_op, G_summaries_op,
                                                                      G_results_op, D_loss_op, D_solver_op,
                                                                      D_summaries_op, D_results_op)
            self.logger.info("Completed TRAINING epoch {}".format(self.curr_epoch))

            # log running time information
            epochs_duration.append(time() - epoch_start_time)
            epoch_mean_time = np.mean(epochs_duration)
            missing_epochs = self.experiment["LEARNING_EPOCHS"] - self.curr_epoch - 1
            training_ETA = timedelta(seconds=epoch_mean_time * missing_epochs)

            self.logger.log_loss("D_loss", "training", self.curr_epoch, D_losses, self.experiment["NUM_ITER_DISCRIMINATOR"])
            self.logger.log_loss("G_loss", "training", self.curr_epoch, G_losses, self.experiment["NUM_ITER_GENERATOR"])
            self.logger.log_epoch_info("training", self.curr_epoch, epoch_results, self.experiment["NUM_ITER_GENERATOR"])
            self.logger.info("Training ETA: {}".format(str(training_ETA)))

            # save evaluation images to tensorboard and to file if necessary
            if self.logger.do_plot_samples(self.curr_epoch):
                self._generate_evaluation_images()

            self.curr_epoch = self.session.run(self.increment_curr_epoch_op)

            if self.logger.do_validate_network(self.curr_epoch):
                self._validate(G_loss_op, G_solver_op, G_summaries_op, G_results_op, D_loss_op, D_solver_op,
                               D_summaries_op, D_results_op)

            if self.logger.do_test_network(self.curr_epoch):
                self._test(G_loss_op, G_solver_op, G_summaries_op, G_results_op, D_loss_op, D_solver_op, D_summaries_op,
                           D_results_op)

            # save training checkpoint if necessary
            if self.logger.do_store_checkpoint(self.curr_epoch):
                self._store_session()

        self.logger.info("Reached the maximum epochs number: training stopped")

        self._store_session()  # always store final checkpoint
        if self.experiment["FINALLY_COMPRESS"]:
            msg = "Compressing {} folder..."
            for f in [self.experiment["OUTPUT_FOLDER"],
                    self.experiment["CHECKPOINT_FOLDER"],
                    self.experiment["TB_EXPERIMENT_FOLDER"]]:
                self.logger.debug(msg.format(f))
                compress_folder(f)

    def _run_train_epoch(self, G_loss_op, G_solver_op, G_summaries_op, G_results_op, D_loss_op, D_solver_op,
                         D_summaries_op, D_results_op):
        """
        run an entire training epoch, collecting, for each batch, the losses of the generator and
        the discriminator. moreover, results are collected in a dict, which keys indicate the measure that
        has been calculated.
        :return: list of losses for the generator and the discriminator and a dict containing the results
        """
        D_losses = []
        G_losses = []
        mbs_duration = []
        results = {}

        for start_idx in range(0, len(self.training_data), self.experiment["BATCH_SIZE"]):

            mb_start_time = time()
            batch_indices = self.training_batches_indices[
                            start_idx:min(start_idx + self.experiment["BATCH_SIZE"], len(self.training_data))]
            training_batch = self.training_data[batch_indices]
            if batch_indices.shape[0] != self.experiment["BATCH_SIZE"]:
                continue
            # collect summaries periodically, not after every batch
            if self.logger.do_log_summaries(start_idx, self.experiment["BATCH_SIZE"], len(self.training_data)):
                current_D_summaries_op = D_summaries_op
                curren_G_summaries_op = G_summaries_op
            else:
                # define a TF "no-op" that can be handled by the
                # None-safe tb_writer
                curren_G_summaries_op = current_D_summaries_op = []

            # each batch can be used to train the discriminator and the
            # generator a different number of times (NUM_ITER_DISCRIMINATOR vs
            # NUM_ITER_GENERATOR)
            ops = [D_loss_op, D_solver_op, current_D_summaries_op, D_results_op]
            for i in range(self.experiment["NUM_ITER_DISCRIMINATOR"]):
                D_loss_mb, D_results = self._discriminator_step(ops, training_batch, batch_indices, "training")
                # save the mini-batch loss
                D_losses.append(D_loss_mb)
                d_outs = dict((k, np.asarray(v)) for k, v in D_results.items())
                # add results to the results dict: values of measures with the same name are organized as lists
                update_dict_of_lists(results, **d_outs)

            ops = [G_loss_op, G_solver_op, curren_G_summaries_op, G_results_op]
            for i in range(self.experiment["NUM_ITER_GENERATOR"]):
                G_loss_mb, G_results = self._generator_step(ops, training_batch, batch_indices, "training")
                # save the mini-batch loss
                G_losses.append(G_loss_mb)
                g_outs = dict((k, np.asarray(v)) for k, v in G_results.items())
                # add results to the results dict: values of measures with the same name are organized as lists
                update_dict_of_lists(results, **g_outs)

            # log running time information if needed
            mbs_duration.append(time() - mb_start_time)
            self.logger.log_epoch_progress("training", self.curr_epoch, start_idx, self.experiment["BATCH_SIZE"],
                                           len(self.training_data),
                                           mbs_duration)

        return D_losses, G_losses, results


    def _run_train_epoch_wgan(self, G_loss_op, G_solver_op, G_summaries_op, G_results_op, D_loss_op, D_solver_op,
                         D_summaries_op, D_results_op):
        """
        run an entire training epoch using the wgan approach, collecting, for each batch, the losses of the generator and
        the discriminator. moreover, results are collected in a dict, which keys indicate the measure that
        has been calculated.
        :return: list of losses for the generator and the discriminator and a dict containing the results
        """
        D_losses = []
        G_losses = []
        mbs_duration = []
        results = {}

        # the total number of batches (comprehensive of the last that may be smaller)
        number_of_batches = math.ceil(len(self.training_data) / self.experiment["BATCH_SIZE"])
        self.logger.info("Batches in this epoch: {}".format(number_of_batches))

        # allow usage by the generator loop
        batch_indices = None
        training_batch = None

        actual_batch = 0
        while actual_batch < number_of_batches:
            # each batch can be used to train the discriminator and the
            # generator a different number of times, to implement the WGAN specification

            ################
            #### CRITIC ####
            ################
            discriminator_iteration = 0
            while discriminator_iteration < self.experiment["NUM_ITER_DISCRIMINATOR"] and actual_batch < number_of_batches:
                # get the first element of the batch
                start_idx = actual_batch * self.experiment["BATCH_SIZE"]

                mb_start_time = time()
                batch_indices = self.training_batches_indices[start_idx:min(start_idx + self.experiment["BATCH_SIZE"], len(self.training_data))]
                training_batch = self.training_data[batch_indices]

                # collect summaries periodically, not after every batch
                if self.logger.do_log_summaries(start_idx, self.experiment["BATCH_SIZE"], len(self.training_data)):
                    current_D_summaries_op = D_summaries_op
                else:
                    # define a TF "no-op" that can be handled by the
                    # None-safe tb_writer
                    current_D_summaries_op = []

                # define operations that should be handled by the discriminator
                ops = [D_loss_op, D_solver_op, current_D_summaries_op, D_results_op]
                # do the discriminator step
                D_loss_mb, D_results = self._discriminator_step(ops, training_batch, batch_indices, "training")
                # save the mini-batch loss
                D_losses.append(D_loss_mb)
                d_outs = dict((k, np.asarray(v)) for k, v in D_results.items())
                # add results to the results dict: values of measures with the same name are organized as lists
                update_dict_of_lists(results, **d_outs)

                # log running time information if needed
                mbs_duration.append(time() - mb_start_time)
                self.logger.log_epoch_progress("DISCRIMINATOR_STEP", self.curr_epoch, start_idx, self.experiment["BATCH_SIZE"],
                                           len(self.training_data),
                                           mbs_duration)
                actual_batch += 1
                discriminator_iteration += 1


            ###################
            #### GENERATOR ####
            ###################
            generator_iteration = 0
            while generator_iteration < self.experiment["NUM_ITER_GENERATOR"]:
                # collect always summaries on the generator
                ops = [G_loss_op, G_solver_op, G_summaries_op, G_results_op]

                # feed generator with data, but will be eventually used by computables and not by the generator itself
                mb_start_time = time()
                # do the generator step
                G_loss_mb, G_results = self._generator_step(ops, training_batch, batch_indices, "training")
                # save the mini-batch loss
                G_losses.append(G_loss_mb)
                g_outs = dict((k, np.asarray(v)) for k, v in G_results.items())
                # add results to the results dict: values of measures with the same name are organized as lists
                update_dict_of_lists(results, **g_outs)

                # log running time information if needed
                mbs_duration.append(time() - mb_start_time)
                self.logger.log_epoch_progress("GENERATOR_STEP", self.curr_epoch, start_idx, self.experiment["BATCH_SIZE"],
                                           len(self.training_data),
                                           mbs_duration)
                # do not update batch count because it has not been consumes by the generator!
                generator_iteration += 1


        return D_losses, G_losses, results

    def _validate(self, G_loss_op, G_solver_op, G_summaries_op, G_results_op, D_loss_op, D_solver_op,
                  D_summaries_op, D_results_op):
        """
        this function is similar to _train, but runs the discriminator and the generator on the
        validation set without updating their weights for one epoch
        """
        if len(self.validation_data) == 0:
            self.logger.info("No test data available, skipping validation.")
            return

        # only one epoch is run: no need to compute ETA
        self.logger.info("Starting VALIDATION epoch")
        self.logger.log_random_states("Start of validation epoch")
        D_losses, G_losses, epoch_results = self._run_validation_epoch(G_loss_op, G_solver_op,
                                                                       G_summaries_op,
                                                                       G_results_op, D_loss_op,
                                                                       D_solver_op,
                                                                       D_summaries_op,
                                                                       D_results_op)
        self.logger.info("Completed VALIDATION epoch")

        self.logger.log_loss("D_loss", "validation", self.curr_epoch, D_losses, self.experiment["NUM_ITER_DISCRIMINATOR"])
        self.logger.log_loss("G_loss", "validation", self.curr_epoch, G_losses, self.experiment["NUM_ITER_GENERATOR"])
        self.logger.log_epoch_info("validation", self.curr_epoch, epoch_results, self.experiment["NUM_ITER_GENERATOR"])

    def _run_validation_epoch(self, G_loss_op, G_solver_op, G_summaries_op, G_results_op, D_loss_op, D_solver_op,
                              D_summaries_op, D_results_op):
        """
        this function is similar to _run_train_epoch, but runs the discriminator and generator on
        the validation set without updating their weights
        :return: list of losses for the generator and the discriminator, a dict containing the results and a
        list of errors on the constraints
        """
        # define operations to run, note that we are not passing solvers

        D_ops = [D_loss_op, []]
        G_ops = [G_loss_op, []]

        D_losses = []
        G_losses = []
        mbs_duration = []
        results = {}

        # do not collect summaries for this epoch
        # define a TF "no-op" that can be handled by the None-safe tb_writer
        D_summaries_op = G_summaries_op = [[]]

        for start_idx in range(0, len(self.validation_data) - self.experiment["BATCH_SIZE"] + 1,
                               self.experiment["BATCH_SIZE"]):
            mb_start_time = time()

            batch_indices = np.arange(start_idx,
                                      min(start_idx + self.experiment["BATCH_SIZE"], len(self.validation_data)))
            validation_batch = self.validation_data[batch_indices]

            ops = D_ops + D_summaries_op + [D_results_op]

            D_loss_mb, D_results = self._discriminator_step(ops, validation_batch, batch_indices, "validation")
            D_losses.append(D_loss_mb)
            d_outs = dict((k, np.asarray(v)) for k, v in D_results.items())
            update_dict_of_lists(results, **d_outs)

            ops = G_ops + G_summaries_op + [G_results_op]
            G_loss_mb, G_results = self._generator_step(ops, validation_batch, batch_indices, "validation")
            G_losses.append(G_loss_mb)
            g_outs = dict((k, np.asarray(v)) for k, v in G_results.items())
            update_dict_of_lists(results, **g_outs)

            # log running time information if needed
            mbs_duration.append(time() - mb_start_time)
            self.logger.log_epoch_progress("validation", self.curr_epoch, start_idx, self.experiment["BATCH_SIZE"],
                                           len(self.validation_data),
                                           mbs_duration)

        return D_losses, G_losses, results

    def _test(self, G_loss_op, G_solver_op, G_summaries_op, G_results_op, D_loss_op, D_solver_op, D_summaries_op,
              D_results_op, seed=None):
        """
        this function is similar to _train, but only runs the discriminator
        on the test set without updating its weights for one epoch
        :param seed:
        """
        if len(self.test_data) == 0:
            self.logger.info("No test data available, skipping test.")
            return

        if seed:
            set_seeds(seed)

        # only one epoch is run: no need to compute ETA
        self.logger.info("Starting TEST epoch")
        self.logger.log_random_states("Start of test epoch")
        D_losses, epoch_results = self._run_test_epoch(G_loss_op, G_solver_op, G_summaries_op,
                                                       G_results_op, D_loss_op, D_solver_op,
                                                       D_summaries_op, D_results_op)

        self.logger.info("Completed TEST epoch")

        self.logger.log_loss("D_loss", "testing", self.curr_epoch, D_losses, self.experiment["NUM_ITER_DISCRIMINATOR"])
        self.logger.log_epoch_info("testing", self.curr_epoch, epoch_results, self.experiment["NUM_ITER_GENERATOR"])

    def _run_test_epoch(self, G_loss_op, G_solver_op, G_summaries_op, G_results_op, D_loss_op, D_solver_op,
                        D_summaries_op, D_results_op):
        """
        perform a single-epoch test
        :return: the loss of the discriminator
        """
        # define operations to run, note that we are not passing the solver, since we are testing
        D_ops = [D_loss_op, []]

        # this function is similar to _run_train_epoch, but only runs the
        # discriminator on the test set without updating its weights
        D_losses = []
        mbs_duration = []
        results = {}

        # do not collect summaries for this epoch
        # define a TF "no-op" that can be handled by the None-safe tb_writer
        D_summaries_op = [[]]
        ops = D_ops + D_summaries_op + [D_results_op]

        for start_idx in range(0, len(self.test_data) - self.experiment["BATCH_SIZE"] + 1,
                               self.experiment["BATCH_SIZE"]):
            mb_start_time = time()

            batch_indices = np.arange(start_idx, min(start_idx + self.experiment["BATCH_SIZE"], len(self.test_data)))
            test_batch = self.test_data[batch_indices]

            D_loss_mb, D_results = self._discriminator_step(ops, test_batch, batch_indices, "testing")
            D_losses.append(D_loss_mb)
            d_outs = dict((k, np.asarray(v)) for k, v in D_results.items())
            update_dict_of_lists(results, **d_outs)

            # log running time information if needed
            mbs_duration.append(time() - mb_start_time)
            self.logger.log_epoch_progress("testing", self.curr_epoch, start_idx, self.experiment["BATCH_SIZE"],
                                           len(self.test_data),
                                           mbs_duration)

        return D_losses, results

    def _generate_input_z_vector(self, l):
        """
        generate l random vectors that will be used as random noise by the generator
        :param l: the number of z-verctors
        :return: a bidimensional random vector of size [l, Z_DIM]
        """
        return np.random.normal(loc=0, scale=1.0, size=(l, self.experiment["Z_DIM"]))
        #return np.random.rand(l, self.experiment["Z_DIM"])

    def _generate_evaluation_z_vector(self, experiment):
        if experiment["EVAL_NOISE_SEED"]:
            msg = "Sampling eval noise with custom seed: {}"
            self.logger.info(msg.format(experiment["EVAL_NOISE_SEED"]))
            old_random_state = np.random.get_state()
            np.random.seed(experiment["EVAL_NOISE_SEED"])
            eval_z = self._generate_input_z_vector(experiment["EVAL_SAMPLES"])
            np.random.set_state(old_random_state)
        else:
            eval_z = self._generate_input_z_vector(experiment["EVAL_SAMPLES"])
        return eval_z
