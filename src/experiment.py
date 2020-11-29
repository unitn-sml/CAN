import json
import os
import numpy as np

import utils.utils_common as utils_common

"""
This class covers everything that characterizes an experiment/run, once an instance
has been created, it should considered read only.
Adding new functionalities or fields after the creation of an instance is to be avoided.

The class works in this way:
1) it imports public functions from the provided modules
2) given the experiment file, it reads parameters that are common to every experiment
3) based on the DATASET_TYPE parameter, it calls a dataset specific loader which given the input experiment file (parsed
as a dict) will take care of reading the necessary and set specific parameters
4) it adds some internal parameters needed for logs, checkpoints etc. storage
"""

# these imports are related to stuff that is going to be picked up
# by the function collector, when you write new imports make sure to avoid
# overshadowing

# generators
from polygons_floor import generators as polygons_floor_generators
from level_generation import generators as level_generators
from semantic_loss import generators as semantic_loss_generators
from semantic_loss import discriminators as semantic_loss_discriminators

# discriminators
from polygons_floor import discriminators as polygons_floor_discriminators
from level_generation import discriminators as level_discriminators

# generic architectures
from level_generation import architectures as level_architectures

# solvers
from polygons_floor import solvers as polygons_floor_solvers
from level_generation import solvers as level_solvers

# losses
import losses as common_losses
from polygons_floor import losses as polygons_floor_losses
from level_generation import losses as level_losses
from fuzzy import losses as fuzzy_losses
from semantic_loss import losses as semantic_losses

# trainer or evaluator
import trainer
import evaluator

# computables
from computables import Computable
from polygons_floor import computables as polygons_floor_computables
from level_generation import computables as level_computables
from semantic_loss import computables as semantic_loss_computables

# statistics
import statistics as common_statistics
from base_layers import Statistic
from polygons_floor import statistics as polygons_floor_statistics
from level_generation import statistics as level_statistics
from semantic_loss import statistics as semantic_statistics
from fuzzy import statistics as fuzzy_statistics

# fuzzy logic circuits
from fuzzy.lyrics import fuzzy as fuzzy_circuits

class Experiment:
    generators_modules = [polygons_floor_generators, level_generators, semantic_loss_generators]
    discriminators_modules = [polygons_floor_discriminators, level_discriminators, semantic_loss_discriminators]
    computables_modules = [polygons_floor_computables, level_computables, semantic_loss_computables]
    solvers_modules = [polygons_floor_solvers, level_solvers]
    losses_modules = [common_losses, polygons_floor_losses, level_losses, semantic_losses, fuzzy_losses]
    statistics_modules = [common_statistics, polygons_floor_statistics, level_statistics, semantic_statistics, fuzzy_statistics]
    architectures_modules = [level_architectures]
    circuits_modules = [fuzzy_circuits]

    def __init__(self):
        """
        retrieve list of names for generators, discriminators, solvers and losses, computables
        """
        #####################################################################################
        # import needed functions from provided modules
        self.generators = Experiment.__import_function_and_classes_from_modules(Experiment.generators_modules)
        self.discriminators = Experiment.__import_function_and_classes_from_modules(Experiment.discriminators_modules)
        self.solvers = Experiment.__import_function_and_classes_from_modules(Experiment.solvers_modules)
        self.losses = Experiment.__import_function_and_classes_from_modules(Experiment.losses_modules)
        self.computables = Experiment.__import_function_and_classes_from_modules(Experiment.computables_modules)
        self.statistics = Experiment.__import_function_and_classes_from_modules(Experiment.statistics_modules)
        self.architectures = Experiment.__import_function_and_classes_from_modules(Experiment.architectures_modules)
        self.circuits = Experiment.__import_function_and_classes_from_modules(Experiment.circuits_modules)

    def read(self, config_file_path, DEBUG):
        """
        read the configuration parameters from the file at config_file_path. the configuration parameters
        that are not required will be ignored. a missing mandatory parameter will raise an error.
        :param config_file_path: the path of the configuration file
        :param DEBUG: if true, execute the experiment in DEBUG mode
        """
        #####################################################################################
        # import configuration file and import required settings (extra stuff will be ignored)
        with open(config_file_path, "r") as conf_f:
            config_dict = json.load(conf_f)
            print("Deserialized JSON: {}".format(json.dumps(config_dict, indent=4)))

        # configurations parameters which are common to all experiments
        common_config = Experiment.__load_common_config(config_dict, self.generators, self.discriminators, self.architectures,
                                                        self.solvers, self.losses, self.computables, self.statistics, self.circuits)

        """
        now that we have the common parameters, use ad hoc loader functions to parse specific parameters
        depending on different kind of dataset
        here you should also set a custom data loader function to use to load data in the way you prefer
        here you should also set your data plotter if you have any, it will be used to plot generated data
        if you have no way of generating images from your data, or you are not interested in doing so,
        plot_data should simply pass
        """
        specific_config, get_dataset, plot_data = Experiment.__load_specific_config(config_file_path, config_dict,
                                                                                    DEBUG, common_config,
                                                                                    self.generators,
                                                                                    self.discriminators,
                                                                                    self.architectures,
                                                                                    self.solvers, self.losses,
                                                                                    self.computables, self.statistics)

        # the specific_config must have a way that defines the dataset_name based on information contained
        # in the config_file, common_config, and its own parameters (needed for internal configuration settings, see
        # __set_internal_config)

        # now we are basically done with collecting everything we need for the experiment
        __experiment_config = utils_common.safe_merge(common_config, specific_config)
        __experiment_config["DEBUG"] = DEBUG
        self.__experiment_config = utils_common.make_read_only(__experiment_config)
        self.get_dataset = get_dataset
        self.plot_data = plot_data

        # this dict contains paths and internal settings needed when storing/loading models, logging, etc.
        __internal_config = Experiment.__set_internal_config(self.__experiment_config, config_file_path, DEBUG)
        self.__internal_config = utils_common.make_read_only(__internal_config)

        """
        check if the experiment_config and internal_config would merge, to make sure that no field appears
        in both dicts, to avoid unexpected behaviours, however we will keep these two dicts separated just to
        have a clear division of the two, internally
        """
        utils_common.safe_merge(self.__experiment_config, self.__internal_config)

    def run(self, tf_session_config, complete_logging):
        """
        run the experiment. extract the datasets and start the trainer with this experiment as parameter.
        :param tf_session_config: the session configuration object
        :param complete_logging: If seed logging should continue during training (costly).
        """
        logger = self["LOGGER"]

        logger.info("Loading dataset...")
        assert self.get_dataset is not None, "You should provide a get_dataset function to load the datasets"
        self.training_data, self.test_data, self.validation_data = self.get_dataset()

        if not isinstance(self.training_data, list):
            self.training_data = self.training_data.astype(np.float32)
        if not isinstance(self.validation_data, list):
            self.validation_data = self.validation_data.astype(np.float32)
        if not isinstance(self.test_data, list):
            self.test_data = self.test_data.astype(np.float32)

        msg = "Samples: {} training, {} test, {} validation\n"
        logger.info(msg.format(len(self.training_data), len(self.test_data), len(self.validation_data)))

        trainer.Trainer(self, tf_session_config, complete_logging)

    def evaluation(self, tf_session_config, complete_logging, number_of_samples):
        """
        take some samples from the experiment.
        :param tf_session_config: the session configuration object
        :param number_of_samples: number of samples that should be generated.
        """

        evaluator.Evaluator(self, tf_session_config, complete_logging, number_of_samples)

    def __getitem__(self, item):
        """
        Access experiment as a dict, both the experiment settings and the internal settings.
        :param item: the item key
        :return: the value of the key
        """
        if item in self.__experiment_config:
            return self.__experiment_config[item]
        elif item in self.__internal_config:
            return self.__internal_config[item]
        else:
            raise KeyError("%s is not part of the experiment configuration (nor the internal one)" % item)

    def __contains__(self, item):
        return item in self.__experiment_config or item in self.__internal_config

    @staticmethod
    def __import_function_and_classes_from_modules(modules_list):
        """
        Given a list of modules, get all public functions and classes and store them into
        a dict mapping their name to the function. Return this dict.
        Names of functions and classes must not clash.
        :return: A dict mapping all public function and classes names of the provided modules to the
        relative functions/classes.
        """
        # for each module, import its functions and classes and merge them in the same dict
        all_function_and_classes = dict()

        for module in modules_list:
            tmp_dict = utils_common.get_module_functions(module.__name__)
            all_function_and_classes = utils_common.safe_merge(all_function_and_classes, tmp_dict)

            tmp_dict = utils_common.get_module_classes(module.__name__)
            all_function_and_classes = utils_common.safe_merge(all_function_and_classes, tmp_dict)

        return all_function_and_classes

    @staticmethod
    def __load_common_config(config_dict, generators, discriminators, architectures, solvers, losses, computables, statistics, circuits):
        """
        Given a config dict obtained by parsing the experiment json file, read the parameters which are required by
        all experiments, assert their correctness, and return a read only dict containing them.
        Parameters not required are ignored.

        :param config_dict: Dict mapping parameters to their values, from the config json file.
        :param generators: Dict mapping generator names to functions constructing a GAN generator.
        :param discriminators: Dict mapping discriminator names to functions constructing a GAN discriminator.
        :param solvers: Dict mapping optimizer names to functions returning a tf optimizer.
        :param losses: Dict mapping losses names to functions returning tf losses.
        :param computables: Dict mapping computables names to computable classes (see documentation).
        :param statistics: Dict mapping statistics names to statistics classes.
        :param circuits: Dict mapping fuzzy logic circuits to their classes
        :return: Dict of parameters which are common to all experiments.
        """
        res_dict = {}

        res_dict["DATASET_TYPE"] = config_dict["DATASET_TYPE"]
        assert isinstance(res_dict["DATASET_TYPE"], str)

        res_dict["TRAINING_TEST_VALIDATION_SPLITS"] = config_dict["TRAINING_TEST_VALIDATION_SPLITS"]
        assert isinstance(res_dict["TRAINING_TEST_VALIDATION_SPLITS"], list)
        assert len(res_dict["TRAINING_TEST_VALIDATION_SPLITS"]) == 3
        for split in res_dict["TRAINING_TEST_VALIDATION_SPLITS"]:
            #assert isinstance(split, (int, float))
            assert isinstance(split, int)
            assert split >= 0
        #at the moment, allow only integer splits
        #assert len(set([type(split) for split in res_dict["TRAINING_TEST_VALIDATION_SPLITS"]])) == 1

        if "SHAPE" in config_dict:
            res_dict["SHAPE"] = config_dict.get("SHAPE")
            assert isinstance(res_dict["SHAPE"], list)
            assert len(res_dict["SHAPE"]) > 0
            for dimension in res_dict["SHAPE"]:
                assert isinstance(dimension, int)
                assert dimension > 0

        # non required param defaulting to 9999 to have retro compatibility
        res_dict["TRAINING_SEED"] = config_dict.get("TRAINING_SEED", 9999)
        assert isinstance(res_dict["TRAINING_SEED"], int)

        res_dict["ANN_SEED"] = config_dict["ANN_SEED"]
        assert isinstance(res_dict["ANN_SEED"], int)

        error_msg = "Can't find %s among:\n%s"
        res_dict["GENERATOR"] = config_dict["GENERATOR"]
        assert res_dict["GENERATOR"] in generators, error_msg % (res_dict["GENERATOR"], list(generators))
        res_dict["GENERATOR"] = generators[res_dict["GENERATOR"]]

        res_dict["DISCRIMINATOR"] = config_dict["DISCRIMINATOR"]
        assert res_dict["DISCRIMINATOR"] in discriminators, error_msg % (
            res_dict["DISCRIMINATOR"], list(discriminators))
        res_dict["DISCRIMINATOR"] = discriminators[res_dict["DISCRIMINATOR"]]

        # G_loss check
        ################
        g_losses = config_dict["GENERATOR_LOSS"]
        # make the single loss as a list if its just a single loss
        g_losses = [g_losses] if isinstance(g_losses, str) else g_losses
        assert isinstance(g_losses, list)

        # either those are all names (no weight associated with each loss) or all pairs (name, weight)
        all_strings = all([isinstance(item, str) for item in g_losses])
        all_pairs = all(
            [(isinstance(item, list) and (len(item) == 2) and isinstance(item[0], str) and isinstance(item[1], (int,float)))
             for item in g_losses])
        assert all_strings or all_pairs, "Losses should either contain a list of names or a list of (name, weight) " \
                                         "pairs. "
        # if those all strings associate to them a weight of 1.
        if all_strings:
            g_losses = [(item, 1.) for item in g_losses]
        # for each loss check that its unique and thats in the possible losses
        g_losses_names = [pair[0] for pair in g_losses]
        assert len(set(g_losses_names)) == len(g_losses_names), \
            "No duplicate generator losses are allowed, if you want to give more importance to a loss," \
            "give it a larger weight"
        for name in g_losses_names:
            assert name in losses, error_msg % (name, list(losses))
        g_losses = [(losses[name], weight) for name, weight in g_losses]
        res_dict["GENERATOR_LOSS"] = g_losses

        # D_loss check
        ################
        d_losses = config_dict["DISCRIMINATOR_LOSS"]
        # make the single loss as a list if its just a single loss
        d_losses = [d_losses] if isinstance(d_losses, str) else d_losses
        assert isinstance(d_losses, list)

        # either those are all names (no weight associated with each loss) or all pairs (name, weight)
        all_strings = all([isinstance(item, str) for item in d_losses])
        all_pairs = all(
            [(isinstance(item, list) and (len(item) == 2) and isinstance(item[0], str) and isinstance(item[1], float))
             for item in d_losses])
        assert all_strings or all_pairs, "Losses should either contain a list of names or a list of (name, weight) " \
                                         "pairs. "
        # if those all strings associate to them a weight of 1.
        if all_strings:
            d_losses = [(item, 1.) for item in d_losses]
        # for each loss check that its unique and thats in the possible losses
        d_losses_names = [pair[0] for pair in d_losses]
        assert len(set(d_losses_names)) == len(d_losses_names), \
            "No duplicate discriminator losses are allowed, if you want to give more importance to a loss," \
            "give it a larger weight"
        for name in d_losses_names:
            assert name in losses, error_msg % (name, list(losses))
        d_losses = [(losses[name], weight) for name, weight in d_losses]
        res_dict["DISCRIMINATOR_LOSS"] = d_losses

        res_dict["BATCH_SIZE"] = config_dict["BATCH_SIZE"]
        assert res_dict["BATCH_SIZE"] > 0

        res_dict["NUM_BGAN_SAMPLES"] = config_dict.get("NUM_BGAN_SAMPLES", None)
        numbgan = res_dict["NUM_BGAN_SAMPLES"]
        assert numbgan is None or (isinstance(numbgan, int) and numbgan > 0)

        #######################
        # learning rates stuff

        common_lr = config_dict.get("LEARNING_RATE", None)
        generator_lr = config_dict.get("GENERATOR_LEARNING_RATE", None)
        discriminator_lr = config_dict.get("DISCRIMINATOR_LEARNING_RATE", None)
        if common_lr is None:
            msg = "If you do not provide a common learning rate 'LEARNING_RATE' then you have to provide " \
                  "one learning rate for the generator and discriminator, 'GENERATOR_LEARNING_RATE' and " \
                  "'DISCRIMINATOR_LEARNING_RATE'."
            assert not (generator_lr is None) and not (discriminator_lr is None), msg
            assert isinstance(generator_lr, float)
            assert isinstance(discriminator_lr, float)
            res_dict["GENERATOR_LEARNING_RATE"] = generator_lr
            res_dict["DISCRIMINATOR_LEARNING_RATE"] = discriminator_lr
        else:
            msg = "If you provide a common learning rate 'LEARNING_RATE' then you should not provide " \
                  "a specific learning rate for the generator and discriminator 'GENERATOR_LEARNING_RATE' " \
                  "and 'DISCRIMINATOR_LEARNING_RATE'."
            assert generator_lr is None and discriminator_lr is None, msg
            assert isinstance(common_lr, float)
            res_dict["GENERATOR_LEARNING_RATE"] = common_lr
            res_dict["DISCRIMINATOR_LEARNING_RATE"] = common_lr
            # for compatibility with old experiments
            res_dict["LEARNING_RATE"] = common_lr

        res_dict["NUM_ITER_GENERATOR"] = config_dict.get("NUM_ITER_GENERATOR", 1)
        assert res_dict["NUM_ITER_GENERATOR"] > 0

        res_dict["NUM_ITER_DISCRIMINATOR"] = config_dict.get("NUM_ITER_DISCRIMINATOR", 1)
        assert res_dict["NUM_ITER_DISCRIMINATOR"] > 0

        # need this to have compatibility with old stuff...
        res_dict["NUM_ITER_GEN"] = res_dict["NUM_ITER_GENERATOR"]
        res_dict["NUM_ITER_DISCR"] = res_dict["NUM_ITER_DISCRIMINATOR"]

        # default leakyness for LeakyReLU layers
        res_dict["LEAKINESS"] = config_dict["LEAKINESS"]
        assert isinstance(res_dict["LEAKINESS"], float)

        # gotta check for both (Z_DIM and z_dim) for backward compatibility
        res_dict["Z_DIM"] = config_dict.get("Z_DIM", config_dict.get("z_dim", None))
        assert isinstance(res_dict["Z_DIM"], int)

        # gotta check for both (H_DIM and h_dim) for backward compatibility
        # TODO: maybe move it to the specific params
        if "H_DIM" in config_dict or "h_dim" in config_dict:
            res_dict["H_DIM"] = config_dict.get("H_DIM", config_dict.get("h_dim", None))
            assert isinstance(res_dict["H_DIM"], int)

        res_dict["LEARNING_EPOCHS"] = config_dict["LEARNING_EPOCHS"]
        assert isinstance(res_dict["LEARNING_EPOCHS"], int)

        res_dict["EVAL_NOISE_SEED"] = config_dict["EVAL_NOISE_SEED"]
        assert isinstance(res_dict["EVAL_NOISE_SEED"], int)

        if "EVAL_SAMPLES" in config_dict:
            res_dict["EVAL_SAMPLES"] = config_dict["EVAL_SAMPLES"]
            assert isinstance(res_dict["EVAL_SAMPLES"], int)
        else:
            res_dict["EVAL_SAMPLES"] = 1

        # discriminator solver
        assert config_dict.get("DISCRIMINATOR_SOLVER") is None or config_dict[
            "DISCRIMINATOR_SOLVER"] in solvers, error_msg % (
            res_dict["DISCRIMINATOR_SOLVER"], list(solvers))
        res_dict["DISCRIMINATOR_SOLVER"] = solvers[config_dict.get("DISCRIMINATOR_SOLVER", "DiscriminatorAdamSolver")]

        # generator solver
        assert config_dict.get("GENERATOR_SOLVER") is None or config_dict["GENERATOR_SOLVER"] in solvers, error_msg % (
            res_dict["GENERATOR_SOLVER"], list(solvers))
        res_dict["GENERATOR_SOLVER"] = solvers[config_dict.get("GENERATOR_SOLVER", "GeneratorAdamSolver")]

        # computables
        res_dict["COMPUTABLES"] = config_dict.get("COMPUTABLES", [])
        assert len(set(res_dict["COMPUTABLES"])) == len(res_dict["COMPUTABLES"])
        assert isinstance(res_dict["COMPUTABLES"], list)
        for computable in res_dict["COMPUTABLES"]:
            assert isinstance(computable, str)
            assert computable in computables, error_msg % (computable, list(computables))
            assert issubclass(computables[computable], Computable)
        res_dict["COMPUTABLES"] = [computables[computable] for computable in res_dict["COMPUTABLES"]]

        # generator statistics
        res_dict["GENERATOR_STATISTICS"] = config_dict.get("GENERATOR_STATISTICS", [])
        if isinstance(res_dict["GENERATOR_STATISTICS"], str):
            res_dict["GENERATOR_STATISTICS"] = [res_dict["GENERATOR_STATISTICS"]]
        assert len(set(res_dict["GENERATOR_STATISTICS"])) == len(res_dict["GENERATOR_STATISTICS"])
        for statistic in res_dict["GENERATOR_STATISTICS"]:
            assert isinstance(statistic, str)
            assert statistic in statistics, error_msg % (statistic, list(statistics))
            assert issubclass(statistics[statistic], Statistic)
        res_dict["GENERATOR_STATISTICS"] = [statistics[statistic] for statistic in res_dict["GENERATOR_STATISTICS"]]

        # discriminator statistics
        res_dict["DISCRIMINATOR_STATISTICS"] = config_dict.get("DISCRIMINATOR_STATISTICS", [])
        if isinstance(res_dict["DISCRIMINATOR_STATISTICS"], str):
            res_dict["DISCRIMINATOR_STATISTICS"] = [res_dict["DISCRIMINATOR_STATISTICS"]]
        assert len(set(res_dict["DISCRIMINATOR_STATISTICS"])) == len(res_dict["DISCRIMINATOR_STATISTICS"])
        for statistic in res_dict["DISCRIMINATOR_STATISTICS"]:
            assert isinstance(statistic, str)
            assert statistic in statistics, error_msg % (statistic, list(statistics))
            assert issubclass(statistics[statistic], Statistic)
        res_dict["DISCRIMINATOR_STATISTICS"] = [statistics[statistic] for statistic in
                                                res_dict["DISCRIMINATOR_STATISTICS"]]

        # start constraints from specific epoch
        res_dict["CONSTRAINTS_FROM_EPOCH"] = config_dict.get("CONSTRAINTS_FROM_EPOCH", None)
        assert res_dict["CONSTRAINTS_FROM_EPOCH"] is None or (
                isinstance(res_dict["CONSTRAINTS_FROM_EPOCH"], int) and res_dict["CONSTRAINTS_FROM_EPOCH"] >= 0)

        # SEMANTIC LOSS STUFF
        # start semantic loss from specific epoch
        res_dict["SEMANTIC_LOSS_FROM_EPOCH"] = config_dict.get("SEMANTIC_LOSS_FROM_EPOCH", None)
        assert res_dict["SEMANTIC_LOSS_FROM_EPOCH"] is None or (
                isinstance(res_dict["SEMANTIC_LOSS_FROM_EPOCH"], int) and res_dict["SEMANTIC_LOSS_FROM_EPOCH"] >= 0)

        # incrementally adjust the semantic loss
        res_dict["SEMANTIC_LOSS_INCREMENTAL"] = config_dict.get("SEMANTIC_LOSS_INCREMENTAL", None)
        assert res_dict["SEMANTIC_LOSS_INCREMENTAL"] is None or (
            isinstance(res_dict["SEMANTIC_LOSS_INCREMENTAL"], bool))

        # if the input we provide to the semantic loss (G_output) is probabilities or raw values/logits
        res_dict["SEMANTIC_LOSS_INPUT_IS_PROBABILITIES"] = config_dict.get("SEMANTIC_LOSS_INPUT_IS_PROBABILITIES", None)
        assert res_dict["SEMANTIC_LOSS_INPUT_IS_PROBABILITIES"] is None or (
            isinstance(res_dict["SEMANTIC_LOSS_INPUT_IS_PROBABILITIES"], bool))
        res_dict["SEMANTIC_LOSS_INPUT_IS_PROBABILITIES"] = bool(res_dict["SEMANTIC_LOSS_INPUT_IS_PROBABILITIES"])

        # FUZZY LOGIC LOSS STUFF
        # start fuzzy logic loss from specific epoch
        res_dict["FUZZY_LOGIC_LOSS_FROM_EPOCH"] = config_dict.get("FUZZY_LOGIC_LOSS_FROM_EPOCH", None)
        assert res_dict["FUZZY_LOGIC_LOSS_FROM_EPOCH"] is None or (
                isinstance(res_dict["FUZZY_LOGIC_LOSS_FROM_EPOCH"], int) and res_dict["FUZZY_LOGIC_LOSS_FROM_EPOCH"] >= 0)

        # incrementally adjust the fuzzy logic loss
        res_dict["FUZZY_LOGIC_LOSS_INCREMENTAL"] = config_dict.get("FUZZY_LOGIC_LOSS_INCREMENTAL", None)
        assert res_dict["FUZZY_LOGIC_LOSS_INCREMENTAL"] is None or (
            isinstance(res_dict["FUZZY_LOGIC_LOSS_INCREMENTAL"], bool))

        # if the input we provide to the fuzzy logic loss (G_output) is probabilities or raw values/logits
        res_dict["FUZZY_LOGIC_LOSS_INPUT_IS_PROBABILITIES"] = config_dict.get("FUZZY_LOGIC_LOSS_INPUT_IS_PROBABILITIES", None)
        assert res_dict["FUZZY_LOGIC_LOSS_INPUT_IS_PROBABILITIES"] is None or (
            isinstance(res_dict["FUZZY_LOGIC_LOSS_INPUT_IS_PROBABILITIES"], bool))
        res_dict["FUZZY_LOGIC_LOSS_INPUT_IS_PROBABILITIES"] = bool(res_dict["FUZZY_LOGIC_LOSS_INPUT_IS_PROBABILITIES"])

        # use dnf or cnf
        res_dict["USE_DNF"] = config_dict.get(
            "USE_DNF", None)
        assert res_dict["USE_DNF"] is None or (
            isinstance(res_dict["USE_DNF"],
                       bool))
        res_dict["USE_DNF"] = bool(
            res_dict["USE_DNF"])

        # extract classes from name
        res_dict["FUZZY_LOGIC_CIRCUIT"] = config_dict.get("FUZZY_LOGIC_CIRCUIT", None)
        assert res_dict["FUZZY_LOGIC_CIRCUIT"] is None or (
            isinstance(res_dict["FUZZY_LOGIC_CIRCUIT"], str))
        
        if res_dict["FUZZY_LOGIC_CIRCUIT"] is None:
            res_dict["FUZZY_LOGIC_CIRCUIT"] = circuits["Lukasiewicz"]
        else:
            assert isinstance(res_dict["FUZZY_LOGIC_CIRCUIT"], str)
            assert res_dict["FUZZY_LOGIC_CIRCUIT"] in circuits, \
                error_msg % (res_dict["FUZZY_LOGIC_CIRCUIT"], list(circuits))
            res_dict["FUZZY_LOGIC_CIRCUIT"] = circuits[res_dict["FUZZY_LOGIC_CIRCUIT"]]
    
        # use soft or hard sampling? default is hard (i.e. 0.6 -> 1)
        res_dict["HARD_SAMPLING"] = config_dict.get("HARD_SAMPLING", True)
        assert isinstance(res_dict["HARD_SAMPLING"], bool)

        # use WGAN for training
        res_dict["WGAN"] = config_dict.get("WGAN", False)
        assert isinstance(res_dict["WGAN"], bool)

        # how many generated/test images to save every epoch
        res_dict["EVAL_IMAGES"] = config_dict.get("EVAL_IMAGES", res_dict["BATCH_SIZE"])
        assert isinstance(res_dict["EVAL_IMAGES"], int) and res_dict["EVAL_IMAGES"] >= 0

        # compress images, tensorboards and models folders at the end of the training?
        res_dict["FINALLY_COMPRESS"] = config_dict.get("FINALLY_COMPRESS", True)
        assert isinstance(res_dict["FINALLY_COMPRESS"], bool)

        return utils_common.make_read_only(res_dict)

    @staticmethod
    def __load_specific_config(config_file_path, config_as_dict, DEBUG, common_config, generators, discriminators,
                               architectures, solvers, losses, computables, statistics):
        """
        Given access to basically everything, for each possible dataset type there should
        be an if/else case which performs three things:
            - provide a dictionary mapping dataset specific parameters (which should be in the provided experiment.json
                to their value
            - provide a get_dataset function which will return training, test, validation set when called with
                no arguments
            - provide a plot_data function which is not expected to return anything and that is called with
                plot_data(samples, numer of epochs), which (might) save stuff to the image output
                directory of your choice (usually out/images/experiment name).
        You have complete freedom on how to provide these three things, given that those are specific to the
        dataset (you must respect their signature, of course).

        :param config_file_path:
        :param config_as_dict:
        :param DEBUG:
        :param common_config:
        :param generators:
        :param discriminators:
        :param solvers:
        :param losses:
        :param computables: Dict mapping computables names to computable classes (see documentation).
        :param statistics: list of statistics
        :return: Dict mapping specific params to their values, get_dataset function, plot_data function.
        """

        # each type of experiment should set "specific_config", "get_dataset", "plot_data" variables in order to
        # have a common interface to load the configurations, to retrieve the datasets and to plot the data

        ####################################
        ######### MNIST STILL LIFE #########
        ####################################
        exp_name = (config_file_path.split(os.sep)[-1]).split(".")[0]

        if "MNIST_STILL_LIFE" in common_config["DATASET_TYPE"]:
            # some stuff we are going to need
            config_file_name = (config_file_path.split("/")[-1]).split(".")[0]
            folder_id = config_file_name + "/"
            output_folder = "out/images/" + folder_id
            shape = config_as_dict["SHAPE"]
            splits = config_as_dict["TRAINING_TEST_VALIDATION_SPLITS"]

            from semantic_loss.loader import get_config_mnist_still_life as get_config_still
            specific_config = get_config_still(config_as_dict)

            from semantic_loss.loader import get_dataset_mnist_still_life as get_dataset_still
            get_dataset = get_dataset_still(common_config["DATASET_TYPE"], splits, shape)

            from semantic_loss.loader import plot_images_mnist_still_life as plot_data_still
            plot_data = plot_data_still(output_folder, shape)
        elif "MNIST_PARITY_CHECK" in common_config["DATASET_TYPE"]:
            # some stuff we are going to need
            config_file_name = (config_file_path.split("/")[-1]).split(".")[0]
            folder_id = config_file_name + "/"
            output_folder = "out/images/" + folder_id
            shape = config_as_dict["SHAPE"]
            splits = config_as_dict["TRAINING_TEST_VALIDATION_SPLITS"]

            from semantic_loss.loader import \
                get_config_mnist_still_life as get_config_still
            specific_config = get_config_still(config_as_dict)

            from semantic_loss.loader import \
                get_dataset_mnist_parity_check as get_dataset_still
            get_dataset = get_dataset_still(common_config["DATASET_TYPE"],
                                            splits, shape)

            from semantic_loss.loader import \
                plot_images_mnist_still_life as plot_data_still
            plot_data = plot_data_still(output_folder, shape)
        elif "MNIST" in common_config["DATASET_TYPE"]:
            # some stuff we are going to need
            config_file_name = (config_file_path.split("/")[-1]).split(".")[0]
            folder_id = config_file_name + "/"
            output_folder = "out/images/" + folder_id
            shape = config_as_dict["SHAPE"]
            splits = config_as_dict["TRAINING_TEST_VALIDATION_SPLITS"]

            from semantic_loss.loader import get_config_mnist_still_life as get_config_still
            specific_config = get_config_still(config_as_dict)

            from semantic_loss.loader import get_dataset_mnist
            get_dataset = get_dataset_mnist(common_config["DATASET_TYPE"], splits, shape)

            from semantic_loss.loader import plot_images_mnist_still_life as plot_data_still
            plot_data = plot_data_still(output_folder, shape)


        ###################################
        #### POLYGONS & FLOOR PLANNING ####
        ###################################
        elif common_config["DATASET_TYPE"] == "polygons" or common_config["DATASET_TYPE"] == "floor_planning":
            """
            Gonna wrap the original experiment class in here and try to emulate what we are doing with
            the other cases.
            """
            # TODO: move all this junk code to a dedicated "loader.py" file in polygons_floor/
            from polygons_floor.experiment import Experiment as poly_Experiment

            wrapped_exp = poly_Experiment(exp_name, config_as_dict, DEBUG)

            # we need to run this now because the old experiment class is setting parameters
            # after instantation, and while running get_dataset()
            training, test, validation = wrapped_exp.get_dataset()

            get_dataset = lambda: (training, test, validation)

            plot_data = wrapped_exp.plot_images

            # get all the stuff that Experiment imported and add them to our specific conf dict, dirty
            # but a time saver, might clean up if i have time
            specific_config = wrapped_exp.__dict__
            new_specific_config = dict()
            for key in specific_config:
                assert isinstance(key, str)
                new_specific_config[key.upper()] = specific_config[key]
            specific_config = new_specific_config

            for key in common_config:
                specific_config.pop(key, None)

            """
            For compatibility reasons we need to keep track of what the internal config of this class would
            set, and remove it from the current dict, make sure to keep this set up to date with the
            __set_internal_config function.
            Note: we cannot first call __set_internal_config because it needs a parameter which should be set
            in the __load_specific_settings function.

            """
            __internal_config_names = {'PY_RANDOM_STATE', 'CHECKPOINT_FOLDER', 'TENSORBOARD_ROOT',
                                       'VALIDATION_STATS_WINDOW',
                                       'CHECKPOINT_FILE', 'DATASETS_FOLDER', 'DATASET_FOLDER', 'LOG_FOLDER', 'LOGGER',
                                       'TB_EXPERIMENT_FOLDER', 'TEST_STATS', 'VALIDATION_STATS', 'OUTPUT_FOLDER',
                                       'PICKLE_FOLDER', 'NP_RANDOM_STATE', 'CONSTRAINED_FLAG'}

            for key in __internal_config_names:
                specific_config.pop(key, None)

            # for now we need this to test/try out multinomial with old experiments
            if len(specific_config["COLOR_MAP"]) == 0:
                specific_config["COLOR_MAP"] = {"placeholder": 1}
            specific_config["CHANNELS"] = len(specific_config["COLOR_MAP"])

            # for retrocompatibility
            if "SHAPE" not in common_config:
                shape = [specific_config["IMG_WIDTH"], specific_config["IMG_HEIGHT"],
                         len(specific_config["COLOR_MAP"])]
                assert shape == specific_config["X_DIM"]
                specific_config["SHAPE"] = shape
            else:
                assert common_config["SHAPE"][-1] == len(specific_config["COLOR_MAP"])
                assert common_config["SHAPE"][0] == specific_config["IMG_WIDTH"]
                assert common_config["SHAPE"][1] == specific_config["IMG_HEIGHT"]
                assert common_config["SHAPE"] == specific_config["X_DIM"]


        ##############################
        ###### LEVEL GENERATION ######
        ##############################
        elif common_config["DATASET_TYPE"] == 'level_generation':

            from level_generation.loader import Loader as LevelLoader
            loader = LevelLoader(config_as_dict, exp_name, architectures)

            specific_config = loader.get_specific_config()
            plot_data = loader.plot_data
            get_dataset = loader.get_dataset


        ###############################
        ## RANDOM FORMULA GENERATION ##
        ###############################
        elif "random_formula" in common_config["DATASET_TYPE"]:
            # some stuff we are going to need
            config_file_name = (config_file_path.split("/")[-1]).split(".")[0]
            folder_id = config_file_name + "/"
            output_folder = "out/images/" + folder_id
            shape = config_as_dict["SHAPE"]
            splits = config_as_dict["TRAINING_TEST_VALIDATION_SPLITS"]

            from semantic_loss.loader import get_config_random_formulas as get_config_formulas
            specific_config = get_config_formulas(config_as_dict)

            from semantic_loss.loader import get_dataset_formulas
            get_dataset = get_dataset_formulas(specific_config["FOLDER_DATASET"], specific_config["FOLDER_DATASET"], splits, shape)

            from semantic_loss.loader import plot_random_formulas as plot_data_formulas
            plot_data = plot_data_formulas(output_folder, shape)

        else:
            # invalid state, raising an error and exiting
            raise ValueError("DATASET_TYPE %s is not recognized" % common_config["DATASET_TYPE"])

        assert specific_config is not None and get_dataset is not None and plot_data is not None, \
            "please provide both specific_config, get_dataset and plot_data." \
            "use empty functions if you don't want to do some things"

        return specific_config, get_dataset, plot_data

    @staticmethod
    def __set_internal_config(experiment_config, config_file_path, DEBUG):
        """
        Return a dict mapping names to paths or other values that are needed internally to coordinate where
        files are saved, validation window, logging, etc.

        :param experiment_config: Dict mapping experiment parameters to their values.
        :param config_file_path: Filename of the experiment configuration json file.
        :param DEBUG: DEBUG level.
        :return A dict mapping internal configuration parameters to their values.
        """
        internal_config = dict()
        internal_config["DATASETS_FOLDER"] = os.path.join("in", "datasets")
        internal_config["DATASET_FOLDER"] = os.path.join(internal_config["DATASETS_FOLDER"], experiment_config["DATASET_NAME"])

        # get from /../name.json -> name.json, then get name
        config_file_name = (config_file_path.split(os.sep)[-1]).split(".")[0]

        folder_id = config_file_name + os.sep
        internal_config["OUTPUT_FOLDER"] = os.path.join("out", "images", folder_id)
        internal_config["CHECKPOINT_FOLDER"] = os.path.join("out", "model_checkpoints", folder_id)
        internal_config["CHECKPOINT_FILE"] = "model"
        internal_config["PICKLE_FOLDER"] = os.path.join(internal_config["DATASETS_FOLDER"], experiment_config["DATASET_NAME"] + "_pickle") + os.sep

        internal_config["TENSORBOARD_ROOT"] = os.path.join("out", "tensorboard") + os.sep
        internal_config["TB_EXPERIMENT_FOLDER"] = os.path.join(internal_config["TENSORBOARD_ROOT"], folder_id)
        internal_config["PY_RANDOM_STATE"] = "py_random_state.pkl"
        internal_config["NP_RANDOM_STATE"] = "np_random_state.pkl"
        internal_config["CONSTRAINED_FLAG"] = "constrained_flag.txt"
        internal_config["LOG_FOLDER"] = os.path.join("out", "log", folder_id)
        os.makedirs(internal_config["LOG_FOLDER"], exist_ok=True)

        internal_config["LOGGER"] = utils_common.set_or_get_logger("general", internal_config[
            "LOG_FOLDER"] + config_file_name + ".log", DEBUG, DEBUG, capacity=10)
        internal_config["SEED_LOGGER"] = utils_common.set_or_get_logger("seed_logger", internal_config[
            "LOG_FOLDER"] + "seeding" + ".log", None, DEBUG, date_fmt="", msg_fmt="%(message)s", capacity=100)
        return internal_config
