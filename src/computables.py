"""
Module containing common computables, which are classes that are init'd with the experiment and data,
and which their compute function will be called before every generator or discriminator step, to help
fill the feed dictionary.
"""
import numpy as np
import logging
import os.path

from utils import utils_common


class Computable:
    """
    Computable base class, other computables must inherit from it and implement the compute method.
    """

    def __init__(self, experiment, training_data, validation_data, test_data, graph_nodes):
        """
        Init the class instance, adding fields or performing pre-run computation.
        An instance of this class may expect a particular node to be present in 'nodes'.
        :param experiment: Instance of the experiment class.
        :param training_data:
        :param validation_data:
        :param test_data:
        :param graph_nodes: Dict mapping names to tf graph_nodes of interest, like placeholders or leafs.
        """
        self.experiment = experiment
        self.logger = self.experiment["LOGGER"]
        self.validation_data = validation_data
        self.training_data = training_data
        self.test_data = test_data
        self.graph_nodes = graph_nodes

        # currently used to decide if the computable should be called when doing evaluation images
        self.needed_for_evaluation_images = False

    def compute(self, feed_dict, shared_dict, curr_epoch=0, real_data_indices=None, generator_step=False,
                step_type="training"):
        """
        Compute the results which are in charge of this class instance, modifying the current feed dict in order to
        fill placeholders needed for the tf session.

        :param feed_dict: Current state of the feed dictionary.
        :param shared_dict: A dictionary that computables can use to communicate with each other and the
            trainer when called (in order), i.e. putting there an already computed G_sample so that the next
            computable to be called won't have to compute it, etc.
        :param curr_epoch: Epoch number.
        :param real_data_indices: Indices corresponding to the real data that may be present in the current_feed_dict,
        named as "X", if present.
        :param generator_step: If this method is being called during a generator or a discriminator step.
        :param step_type: A string naming in what kind of step are we. Value must be either training, validation,
            or testing.
        """
        raise NotImplementedError("This method should be overridden by children")


class ConstraintsComputable(Computable):

    def __init__(self, experiment, training_data, validation_data, test_data, graph_nodes):
        """
        Init the class instance, adding fields and performing constraints computation
        on real data, caching it for future use.
        An instance of this class may expect a particular node to be present in 'nodes'.
        :param experiment: Instance of the experiment class.
        :param training_data:
        :param validation_data:
        :param test_data:
        :param graph_nodes: Dict mapping names to tf graph_nodes of interest, like placeholders or leafs.
        """
        super().__init__(experiment, training_data, validation_data, test_data, graph_nodes)

        cached_constraints = self._compute_real_constraints_caches(training_data, validation_data, test_data)
        self.real_training_constraints_cache = cached_constraints[0]
        self.real_validation_constraints_cache = cached_constraints[1]
        self.real_test_constraints_cache = cached_constraints[2]

        # proxy constraints, for where no constraints have to be computed, we can just use zeros
        self.real_proxy_constraints, self.fake_proxy_constraints = \
            self._proxy_constraints(experiment, len(self.__class__.constraints_names()))

        # init logger and log files peculiar to this logger
        self._init_logging_fields()

        if "C_fake" not in self.graph_nodes:
            self.logger.info(
                "C_fake not found in graph_nodes, %s will not add fake constraints to the feed_dict"
                % self.__class__.__name__)
        if "C_real" not in self.graph_nodes:
            self.logger.info(
                "C_real not found in graph_nodes, %s will not add real constraints to the feed_dict"
                % self.__class__.__name__)
        if "use_constraints" not in self.graph_nodes:
            self.logger.info(
                "use_constraints not found in graph_nodes, %s will not add this boolean to the feed_dict"
                % self.__class__.__name__)

        self.started_constrained_training = False

    def _compute_real_constraints_caches(self, training_data, validation_data, test_data):
        """
        Compute constraints on already available data before starting training, to compute them only once.
        If no constrained training takes place, those constraints are only computed on validation and test data,
        so that how much real data respects constraints can still be logged at validation/test time, and compared
        with fake data.

        :param training_data:
        :param validation_data:
        :param test_data:
        :return: Three elements, computed constraints on real, validation and test data. These elements
        may be None if their relative data is of length 0.
        """

        self.logger.info("Computing constraints on train, validation and test sets before training")
        if (self.experiment["CONSTRAINTS_FROM_EPOCH"] is not None) and (len(training_data) > 0):
            training_constraints = self._constraints_function(training_data)
        else:
            training_constraints = None
        validation_constraints = None if len(validation_data) == 0 else self._constraints_function(validation_data)
        test_constraints = None if len(test_data) == 0 else self._constraints_function(test_data)
        return training_constraints, validation_constraints, test_constraints

    def _proxy_constraints(self, experiment, n_constraints):
        """
        Obtain the proxy constraints to be used when actually computed constraints do not need to be used, for
        example during training if no constrained training takes place.
        Real proxy constraints are of shape [batch size, n_constraints], while fake proxy constraints
        are of shape [samples * batch size, n_constraints], where samples is 1 if NUM_BGAN_SAMPLES is None,
        else it is equal to experiment[NUM_BGAN_SAMPLES].

        :param experiment: Instance of the experiment class.
        :param n_constraints: How many constraints for each data sample should be returned, the second dimension
        of the return real and proxy constraints, which are 2 dimensional and of shape [-1, n_constraints].
        :return: Real and fake data proxy constraints, which are constraints that are not actually computed, all zeros.
        """
        bs = experiment["BATCH_SIZE"]
        numbgan = experiment["NUM_BGAN_SAMPLES"]
        samples = 1 if numbgan is None else numbgan

        real_proxy_constraints = np.zeros([bs, n_constraints], np.float32)
        fake_proxy_constraints = np.zeros([samples * bs, n_constraints], np.float32)
        return real_proxy_constraints, fake_proxy_constraints

    def _compute_fake_constraints(self, graph_nodes, current_feed_dict, shared_dict):
        """
        Given the graph_nodes and the current_feed_dict, compute a G_sample and then compute
        constraints based on that.
        This function will expect G_sample to be in graph_nodes, and that whatever is in current_feed_dict (i.e. z)
        is enough to compute G_sample.

        :param graph_nodes: Dict mapping names to tf graph_nodes of interest, like placeholders or leafs.
        :param current_feed_dict: Current state of the feed dictionary.
        :param shared_dict: A dictionary that computables can use to communicate with each other and the
            trainer when called (in order), i.e. putting there an already computed G_sample so that the next
            computable to be called won't have to compute it, etc.
        :return: Constraint values on G_sample computed starting from the current_feed_dict.
        """
        assert "G_sample" in graph_nodes, "Expected to have G_sample in the graph_nodes, given that we need it " \
                                          "to compute constraints results on fake data. "

        if graph_nodes["G_sample"] in shared_dict:
            # if some computable has already put G_sample there
            generated_data = shared_dict[graph_nodes["G_sample"]]
        else:
            G_sample = graph_nodes["G_sample"]

            # need to get these out if some other computable has already added them,
            # otherwise it will try to feed those values to placeholders, even if those placeholders are not used
            hascfakes = graph_nodes["C_fake"] in current_feed_dict
            hascreal = graph_nodes["C_real"] in current_feed_dict
            C_fake = current_feed_dict.pop(graph_nodes["C_fake"], None)
            C_real = current_feed_dict.pop(graph_nodes["C_real"], None)

            generated_data = G_sample.eval(current_feed_dict)

            # add it to the shared dict so that other may find/use it
            shared_dict[graph_nodes["G_sample"]] = generated_data

            if hascfakes:
                current_feed_dict[graph_nodes["C_fake"]] = C_fake
            if hascreal:
                current_feed_dict[graph_nodes["C_real"]] = C_real

        generated_data = generated_data.reshape([-1] + self.experiment["SHAPE"])
        constraints_values = self._constraints_function(generated_data)
        assert constraints_values.shape[1] == len(self.__class__.constraints_names())
        return constraints_values

    def compute(self, feed_dict, shared_dict, curr_epoch=0, real_data_indices=None, generator_step=False,
                step_type="training"):
        """
        Compute constraints on real and fake data and fill them in the feed_dict as a mapping graph_node[C_real]
        and graph_node[C_fake] to them if C_fake and C_real are present in the graph_nodes.
        Note that if C_real and C_fake are already in the feed_dict, those already present values will be concatenated
        with the current ones, in order to support the usage of multiple ConstraintsComputable classes.
        If C_fake and C_real are not present and the step_type is either validation or testing, constraints
        are computed anyway for logging purposes.
        :param feed_dict: Current state of the feed dictionary.
        :param shared_dict: A dictionary that computables can use to communicate with each other and the
            trainer when called (in order), i.e. putting there an already computed G_sample so that the next
            computable to be called won't have to compute it, etc.
        :param curr_epoch: Epoch number.
        :param real_data_indices: Indices corresponding to the real data that may be present in the current_feed_dict,
        named as "X", if present.
        :param generator_step: If this method is being called during a generator or a discriminator step.
        :param step_type: A string naming in what kind of step are we. Value must be either training, validation,
            or testing.
        """
        constraints_needed_for_training = self.experiment["CONSTRAINTS_FROM_EPOCH"] is not None and curr_epoch >= \
                                          self.experiment["CONSTRAINTS_FROM_EPOCH"]
        assert not (constraints_needed_for_training and (
                "C_fake" not in self.graph_nodes or "C_real" not in self.graph_nodes)), "Constrained training is " \
                                                                                        "active, but C_real or " \
                                                                                        "C_fake are not present " \
                                                                                        "in the graph_nodes, " \
                                                                                        "meaning that the " \
                                                                                        "computation of " \
                                                                                        "constraints at training " \
                                                                                        "time does not make " \
                                                                                        "sense. This is most likely " \
                                                                                        "a discriminator " \
                                                                                        "architecture problem. "

        allowed_steps = ["training", "validation", "testing"]
        assert step_type in allowed_steps

        # log only the first time the constrains are used for training
        if constraints_needed_for_training and not self.started_constrained_training:
            self.started_constrained_training = True
            self.logger.info("Starting Constrained training at epoch %s" % curr_epoch)

        # start with proxy constraints, actually use the computed ones if needed
        fake_constraints = self.fake_proxy_constraints
        real_constraints = self.real_proxy_constraints

        if constraints_needed_for_training or step_type != "training":
            fake_constraints = self._compute_fake_constraints(self.graph_nodes, feed_dict, shared_dict)

            # need to check in case the data was of length 0
            if self.real_training_constraints_cache is None:
                real_constraints = self.real_proxy_constraints
            else:
                cache = self.real_training_constraints_cache
                if step_type == "validation":
                    cache = self.real_validation_constraints_cache
                elif step_type == "testing":
                    cache = self.real_test_constraints_cache
                real_constraints = cache[real_data_indices]

        # log statistics on constraints if we are in evaluation or test phase
        if step_type != "training":
            self._log_constraints_statistics(real_constraints, curr_epoch, "real", step_type)
            self._log_constraints_statistics(fake_constraints, curr_epoch, "fake", step_type)

        # add to feed dict if needed
        # if its in graph_nodes we need to fill it
        if "C_fake" in self.graph_nodes:
            if self.graph_nodes["C_fake"] in feed_dict:
                # already there, concatenate it
                current = feed_dict[self.graph_nodes["C_fake"]]
                assert len(current.shape) == len(fake_constraints.shape)
                assert len(current.shape) == 2
                assert current.shape[0] == fake_constraints.shape[0]
                feed_dict[self.graph_nodes["C_fake"]] = np.concatenate([current, fake_constraints], axis=1)
            else:
                # otherwise just assign it
                feed_dict[self.graph_nodes["C_fake"]] = fake_constraints

        if "C_real" in self.graph_nodes:
            if self.graph_nodes["C_real"] in feed_dict:
                # already there, concatenate it
                current = feed_dict[self.graph_nodes["C_real"]]
                assert len(current.shape) == len(real_constraints.shape)
                assert len(current.shape) == 2
                assert current.shape[0] == real_constraints.shape[0]
                feed_dict[self.graph_nodes["C_real"]] = np.concatenate([current, real_constraints], axis=1)
            else:
                # otherwise just assign it
                feed_dict[self.graph_nodes["C_real"]] = real_constraints

        if "use_constraints" in self.graph_nodes:
            # make sure that if at least 1 ConstraintsComputable needs use_constraints
            # the value remains the same
            feed_dict[self.graph_nodes["use_constraints"]] = constraints_needed_for_training or feed_dict.get(
                self.graph_nodes["use_constraints"], False)

    def _constraints_function(self, data):
        """
        This is the function that effectively computes constraints on given data, must be implemented by the user.
        :param data: Data on which to compute the constraints function.
        :return:
        """
        raise NotImplementedError("This method should be overridden by children")

    @staticmethod
    def constraints_names():
        """
        Returns a list of names, where ith name represent the ith constraint computed
        by the constraint function of this class. Make sure the length of this list
        is equal to the number of constraints computed for each data point.
        :return:
        """
        raise NotImplementedError("This method should be overridden by children")

    def _log_constraints_statistics(self, constraints, epoch, data_type, step_type):
        """
        Store and logs constraints when at the end of an epoch.
        Constraints results are stored for each batch until the end of the current epoch, after that they are
        aggregated, logged and flushed.

        :param constraints: Constraints results for the current batch, shape of [-1, len(constraints_names())]
        :param epoch: Current epoch number.
        :param data_type: Type of constraints we are logging, can be either 'fake' or 'real'.
        :param step_type: Type of step we are logging, can be either 'testing' or 'validation'.
        """
        allowed_data_type = ["real", "fake"]
        allowed_step_type = ["validation", "testing"]
        assert data_type in allowed_data_type
        assert step_type in allowed_step_type

        # get constraints for this epoch related to the data_type and step_type, append the ones related to the new
        # batch
        epoch_constraints = self._constraints_results_current_epoch["_%s_%s" % (step_type, data_type)]
        epoch_constraints.append(constraints)
        expected_epoch_batches = self._expected_test_batches \
            if step_type == "testing" else self._expected_validation_batches
        if len(epoch_constraints) < expected_epoch_batches:
            return

        # if we are at the end of the epoch aggregate and log statistics
        epoch_constraints = np.concatenate(epoch_constraints, axis=0)
        # reset to an empty list for the next epoch
        self._constraints_results_current_epoch["_%s_%s" % (step_type, data_type)] = []

        # everything we need to write the current statistics as the next row of a csv file
        row_to_write = [epoch]

        # number of samples
        num_stats_samples = len(epoch_constraints)

        # for each constraint compute average, std, min, max, percentage of perfect objects
        for j, fn_name in enumerate(self.__class__.constraints_names()):
            # filter errors for the current constraint function
            errors = [err_tuple[j] for err_tuple in epoch_constraints]

            row_to_write.append(np.average(errors))
            row_to_write.append(np.var(errors))
            row_to_write.append(min(errors))
            row_to_write.append(max(errors))

            zero_error = errors.count(0)
            perfect_percentage = 100 * zero_error / num_stats_samples
            row_to_write.append(perfect_percentage)

        # compute the number of perfect objects with respect to all constraints
        if len(self.__class__.constraints_names()) > 0:
            all_zero = len([t for t in epoch_constraints if sum(t) == 0])
            perfect_percentage = 100 * all_zero / num_stats_samples
            row_to_write.append(perfect_percentage)
            msg = "%s for %s constraints, %s data perfect items: %s %%"
            msg = msg % (step_type.upper(), self.__class__.__name__, data_type.upper(), perfect_percentage)
            self.logger.info(msg)

        # write row to csv through the personal class logger related to this step type and data type
        row_to_write = [str(item) for item in row_to_write]
        self._log_to_file(step_type, data_type, row_to_write)

    def _log_to_file(self, step_type, data_type, data):
        """
        Logs the data to the correct file according to the step_type and data_type.
        Files are initialized in a lazy way. Also note that files that already exist will not
        be overwritten but appended to, in order to allow for the correct resuming
        of a session.
        If you start a session anew you are in charge of deleting the old files, that can be
        found in the log directory of this experiment.
        :param data_type: Type of constraints we are logging, can be either 'fake' or 'real'.
        :param step_type: Type of step we are logging, can be either 'testing' or 'validation'.
        :param data:
        """

        logname = "%s_%s_%s" % (self.__class__.__name__, step_type, data_type)
        fname = self.experiment["LOG_FOLDER"] + logname + ".csv"

        # if file does not exist already then we will write header, after that log data
        # avoid writing header two times if the session is restored
        write_header = not os.path.exists(fname)

        # either create logger (and file) or retrieve the logger
        logger = utils_common.set_or_get_logger(logname, fname, console_level=None, file_level=logging.INFO,
                                                msg_fmt="", date_fmt="")
        if write_header:
            logger.info(self._header)
        logger.info(",".join(data))

    def _init_logging_fields(self):
        """
        Init the instance fields required for logging purposes.
        """
        # we will need the expected batches to know when to aggregate everything and compute stats
        _, test_samples, validation_samples = self.experiment["TRAINING_TEST_VALIDATION_SPLITS"]
        self._expected_test_batches = test_samples // self.experiment["BATCH_SIZE"]
        self._expected_validation_baches = validation_samples // self.experiment["BATCH_SIZE"]

        # header of the csv file
        header = ["epoch"]
        for name in self.__class__.constraints_names():
            header.append("E[%s]" % name)
            header.append("Var[%s]" % name)
            header.append("min[%s]" % name)
            header.append("max[%s]" % name)
            header.append("perfect_objects[%s]" % name)
        header.append("perfect_objects[all_constraints]")
        header = ",".join(header)
        self._header = header

        self._constraints_results_current_epoch = dict()
        for f in ["_testing_real", "_testing_fake", "_validation_real", "_validation_fake"]:
            # for each step_type and data_type we will collect results over all batches, then aggregate once for
            # every epoch
            self._constraints_results_current_epoch[f] = []
