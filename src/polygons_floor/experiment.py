import glob
import numpy as np
import os
import json
import random
import scipy.misc
import scipy.ndimage

from inspect import getmembers, ismethod
from itertools import islice
from shutil import unpack_archive
from utils.utils_common import pickle_binary_file, unpickle_binary_file
from utils.utils_tf import escape_snapshot_name as replace_brackets


class Experiment:
    def __init__(self, experiment_file, conf, debug):
        # mandatory parameters: conf["PARAM"]
        # optional parameters: conf.get("PARAM")

        self.dataset_type = conf["DATASET_TYPE"]
        self.img_width = conf["IMG_WIDTH"]
        self.img_height = conf["IMG_HEIGHT"]

        if self.dataset_type == "polygons":
            self._load_polygons_config(conf)
        elif self.dataset_type == "floor_planning":
            self._load_floor_planning_config(conf)
        else:  # "custom" type
            self.binomial = True
            self.dataset_name = conf["DATASET_NAME"]
            self.dataset_loader_fn = conf["LOADER_FUNCTION"]
            self.loader_path = conf["LOADER_PATH"]

        self.training_seed = conf.get("TRAINING_SEED")
        if not self.training_seed:
            self.training_seed = 9999
        self.ann_seed = conf["ANN_SEED"]
        self.generator = conf["GENERATOR"]
        self.discriminator = conf["DISCRIMINATOR"]

        self.batch_size = conf["BATCH_SIZE"]
        self.num_samples = conf.get("NUM_BGAN_SAMPLES", None)
        self.learning_rate = conf.get("LEARNING_RATE", None)
        self.num_iter_gen = conf["NUM_ITER_GENERATOR"]
        self.num_iter_discr = conf["NUM_ITER_DISCRIMINATOR"]
        self.leak = conf["LEAKINESS"]
        self.z_dim = conf["z_dim"]
        self.h_dim = conf["h_dim"]

        self.learning_epochs = conf.get("LEARNING_EPOCHS")

        self.eval_noise_seed = conf.get("EVAL_NOISE_SEED")

        self.constraints_fn_real_data = conf.get("CONSTRAINTS_FN_REAL_DATA")
        self.constraints_fn_fake_data = conf.get("CONSTRAINTS_FN_FAKE_DATA")
        self.constrained_training = conf.get("CONSTRAINED_TRAINING", False)
        self.constraints_from_epoch = conf.get("CONSTRAINTS_FROM_EPOCH")
        self.constraints_type = conf.get("CONSTRAINTS_TYPE")
        self.debug = debug

        if not "DEBUG_CONSTRAINTS" in conf:
            self.debug_constraints = False
        else:
            self.debug_constraints = conf["DEBUG_CONSTRAINTS"]
        if self.constraints_type == "GENERATOR":
            self.normalization_factor = \
                conf["CONSTRAINTS_GENERATOR_NORMALIZATION_FACTOR"]
        self.X_dim = [self.img_width, self.img_height, 1]
        if self.dataset_type == "custom":
            name = "{}_S{}_G{}_D{}_mb{}_ep{}_lr{}_zdim{}_hdim{}"
            self.name = name.format(
                experiment_file, self.ann_seed, self.generator,
                self.discriminator, self.batch_size, self.learning_epochs,
                self.learning_rate, self.z_dim, self.h_dim)
        else:  # "polygons" or "floor_planning" types
            name = "{}_ttv{}_{}_{}_S{}_G{}_D{}_mb{}_ep{}_lr{}_zdim{}_hdim{}"
            self.name = name.format(
                experiment_file, self.dataset_splits[0],
                self.dataset_splits[1], self.dataset_splits[2], self.ann_seed,
                self.generator, self.discriminator, self.batch_size,
                self.learning_epochs, self.learning_rate, self.z_dim,
                self.h_dim)

        self.datasets_folder = "in/datasets/"
        self.dataset_folder = self.datasets_folder + self.dataset_name + "/"
        folder_id = experiment_file + "/"
        self.output_folder = "out/images/" + folder_id
        self.checkpoint_folder = "out/model_checkpoints/" + folder_id
        self.checkpoint_file = "model"
        self.pickle_folder = self.datasets_folder + self.dataset_name + \
                             "_pickle/"
        os.makedirs(self.pickle_folder, exist_ok=True)

        self.tensorboard_root = "out/tensorboard/"
        self.tb_experiment_folder = self.tensorboard_root + folder_id
        self.py_random_state = "py_random_state.pkl"
        self.np_random_state = "np_random_state.pkl"
        self.validation_stats_window = 5
        self.validation_stats = "validation_stats.dill"
        self.test_stats = "test_stats.dill"
        self.constrained_flag = "constrained_flag.txt"
        # since Python's str.__hash__ is non-deterministic across different
        # runs it is necessary to use hashlib

        self.color_map = {}

        self.log_folder = "out/log/"
        os.makedirs(self.log_folder, exist_ok=True)

    def __repr__(self):
        return str(self.__dict__)

    def plot_images(self, samples, epoch, evaluation=False):
        # samples shape: (eval_samples, width, height, channels)
        image_path = self.output_folder + "{}_{}.png"
        # images path
        images_path = os.path.join(self.output_folder,
                                   "evaluation-images" if evaluation else "training-images")
        name_images = images_path + "/evaluation-images-{}-{}.png" if evaluation else "sample-epoch-%d-number-{}.png" % epoch
        if self.binomial:
            for j, image in enumerate(samples):
                image = image.reshape((self.img_width, self.img_height))
                scipy.misc.imsave(image_path.format(
                    str(epoch).zfill(6), str(j).zfill(2)), image)
        else:
            inverse_colormap = {v: k for k, v in self.color_map.items()}
            for j, image in enumerate(samples):
                image = self._one_hot_decode(image, inverse_colormap)
                scipy.misc.imsave(image_path.format(
                    str(epoch).zfill(6), str(j).zfill(2)), image)

        if evaluation:
            # check if folder already exists and eventually create it
            if not os.path.exists(images_path):
                os.mkdir(images_path)
            for j, image in enumerate(samples):
                image = image.reshape((self.img_width, self.img_height))
                scipy.misc.imsave(name_images.format(epoch, j), image)


    def get_dataset(self):
        # train/test/validation splits for specific seed
        files = [f.format(self.ann_seed) for f in [
            "training_{}.pkl", "test_{}.pkl", "validation_{}.pkl"]]
        color_map_file = self.pickle_folder + "color_map_{}.pkl" \
            .format(self.ann_seed)
        if all(map(lambda f: os.path.exists(self.pickle_folder + f), files)):
            # if the dataset is composed only by black and white pixel
            # we can simply load the files,  otherwise we need a colour map
            if self.binomial:
                print("Pickle dataset files found for seed {}. Going "
                      "to load them...".format(self.ann_seed))
                return self._load_dataset_from_files(files)
            if os.path.exists(color_map_file):
                print("Pickle dataset files found for seed {}"
                      " together with color map {}."
                      " Going to load them..."
                      .format(self.ann_seed, color_map_file))
                self.color_map = unpickle_binary_file(color_map_file)
                # update X_dim now that possible output values number is known
                self.X_dim = [self.img_width, self.img_height,
                              len(self.color_map)]
                return self._load_dataset_from_files(files)
            else:
                print("Unable to find color map {}. Going to "
                      "reload dataset..."
                      .format(color_map_file))
        print("Pickle dataset files not found for seed {}. "
              "Creating dataset...".format(self.ann_seed))
        if self.dataset_type == "custom":
            fns = dict(getmembers(self, predicate=ismethod))
            training, test, validation = fns[self.dataset_loader_fn]()
        else:  # "polygons" types
            training, test, validation = self._create_from_images()
        print("Dataset created. Saving pickle dataset files...")
        self._save_dataset_to_files(files, training, test, validation)
        if not self.binomial:
            print("Saving color map file {}...".format(
                color_map_file))
            pickle_binary_file(color_map_file, self.color_map)
        return training, test, validation

    def _load_dataset_from_files(self, files):
        return tuple(unpickle_binary_file(self.pickle_folder + f)
                     for f in files)

    def _save_dataset_to_files(self, files, training, test, validation):
        for split in zip(files, [training, test, validation]):
            pickle_binary_file(self.pickle_folder + split[0], split[1])

    def _create_from_images(self):
        # extract dataset archive if necessary
        if not os.path.exists(self.dataset_folder):
            archive = self.dataset_name + ".zip"
            archive_path = self.datasets_folder + archive
            msg = "Extracting dataset archive {}..."
            print(msg.format(archive_path))
            unpack_archive(archive_path, self.dataset_folder, 'zip')

        msg = "Creating dataset from folder {}..."
        print(msg.format(self.dataset_folder))
        files_names = glob.glob(glob.escape(self.dataset_folder) + "*.png")
        random.seed(self.ann_seed)
        random.shuffle(files_names)  # guarantees consistency for ttv splits
        if self.binomial:
            images_data = self._read_binomial_dataset(files_names)
        else:
            images_data = self._read_multinomial_dataset(files_names)

        it = iter(images_data)
        training, test, validation = (list(islice(it, 0, i))
                                      for i in self.dataset_splits)

        return self._reshape_samples(training) if len(training) > 0 else [], \
               self._reshape_samples(test) if len(test) > 0 else [], \
               self._reshape_samples(validation) if len(validation) > 0 else []

    def _read_binomial_dataset(self, files_names):
        images_data = []
        for image in files_names:
            # read image as greyscale
            image_data = scipy.ndimage.imread(image, flatten=True)
            image_data[image_data == 255] = 1  # binarize (255 -> 1)
            images_data.append(image_data)
        return images_data

    def _read_multinomial_dataset(self, files_names):
        print("Building color map from dataset...")
        # read coloured image
        raw_data = [scipy.ndimage.imread(image) for image in files_names]
        self._build_color_map(raw_data)
        print("Encoding dataset...")
        # update X_dim now that possible output values number is known
        self.X_dim = [self.img_width, self.img_height, len(self.color_map)]
        return [self._one_hot_encode(image) for image in raw_data]

    def _build_color_map(self, raw_data):
        j = 0
        for image_data in raw_data:
            for x in range(image_data.shape[0]):
                for y in range(image_data.shape[1]):
                    value = tuple(image_data[x][y])
                    if value not in self.color_map:
                        self.color_map[value] = j
                        j += 1
        print("Color map: {}".format(self.color_map))

    def _one_hot_encode(self, image_data):
        # from RGB to one-hot enconding for pixel colour
        color_numbers = len(self.color_map.items())
        # image_data shape: (width, height, channels)
        rows = []
        for x in range(image_data.shape[0]):
            col = []
            for y in range(image_data.shape[1]):
                value = tuple(image_data[x][y])
                one_hot_encoding = color_numbers * [0]
                one_hot_encoding[self.color_map[value]] = 1
                col.append(one_hot_encoding)
            rows.append(col)
        return np.asarray(rows)

    def _one_hot_decode(self, image_data, inverse_color_map):
        # TODO CHECK IF IT IS RIGHT
        # image_data shape: (width, height, 1)
        # the value on the last dimension is the index of the class, drawn from
        # the multinomial distribution (i.e. 0 -> 1st class, 1 -> 2nd class,
        # 2 -> 3rd class, ...). In other words, it is the index of the "1" in
        # the one-hot-encoding
        rows = []
        for x in range(image_data.shape[0]):
            col = []
            for y in range(image_data.shape[1]):
                class_position = int(np.where(image_data[x][y] == 1)[0])
                col.append(inverse_color_map[class_position])
            rows.append(col)
        return np.asarray(rows)

    def _reshape_samples(self, samples):
        return np.reshape(np.stack(samples), newshape=(
            -1, self.img_width, self.img_height, max(1, len(self.color_map))))

    def _load_polygons_config(self, conf):
        self.dataset_seed = conf["DATASET_SEED"]
        self.dataset_size = conf["DATASET_SIZE"]
        self.polygons_number = conf["POLYGONS_NUMBER"]
        self.polygons_prob = conf["POLYGONS_PROB"]
        self.area = conf["AREA"]
        self.dataset_splits = tuple(
            conf["TRAINING_TEST_VALIDATION_SPLITS"])

        if not "NUM_COLORS" in conf:
            self.num_colors = 1
        else:
            self.num_colors = conf["NUM_COLORS"]

        if not "BINOMIAL" in conf:
            self.binomial = (self.num_colors < 2)
        else:
            self.binomial = conf["BINOMIAL"]

        if self.binomial and self.num_colors > 1:
            msg = "Can't use the binomial architecture with > 2 values"
            raise ValueError(msg)

        self.suffix_id = conf["SUFFIX_ID"]
        if isinstance(self.area, int):
            self.dataset_name = "{}k_{}x{}_pol{}_area{}_S{}{}".format(
                int(self.dataset_size // 1000), self.img_width,
                self.img_height, replace_brackets(str(self.polygons_prob)),
                self.area, self.dataset_seed, self.suffix_id). \
                replace(" ", "_")
        else:
            self.dataset_name = "{}k_{}x{}_pol{}_area{}_S{}{}".format(
                int(self.dataset_size // 1000), self.img_width,
                self.img_height, replace_brackets(str(self.polygons_prob)),
                replace_brackets(str(self.area)),
                self.dataset_seed, self.suffix_id).replace(" ", "_")

    def _load_floor_planning_config(self, conf):
        self.dataset_seed = conf["DATASET_SEED"]
        self.dataset_size = conf["DATASET_SIZE"]
        self.buildings_prob = conf["BUILDINGS_PROB"]
        self.max_building_per_tree = conf["MAX_BUILDING_PER_TREE"]
        self.images_per_cluster = conf["IMAGES_PER_CLUSTER"]
        self.apt_people = conf["APT_PEOPLE"]
        self.res_people = conf["RES_PEOPLE"]
        self.double_res_people = conf["DOUBLE_RES_PEOPLE"]
        self.apt_area = conf["APT_AREA"]
        self.res_area = conf["RES_AREA"]
        self.double_res_area = conf["DOUBLE_RES_AREA"]
        self.dataset_splits = tuple(
            conf["TRAINING_TEST_VALIDATION_SPLITS"])
        self.binomial = False

        self.suffix_id = conf["SUFFIX_ID"]
        self.dataset_name = "{}k_{}x{}_S{}{}" \
            .format(
            int(self.dataset_size // 1000), self.img_width,
            self.img_height, self.dataset_seed, self.suffix_id) \
            .replace(" ", "_")
        self.problem_path = "in/experiments/polygons_floor/floor_planning.mzn"


def load_experiment_config(experiment_path, debug=False):
    with open(experiment_path, "r") as conf_f:
        filename = experiment_path.split(os.sep)[-1]
        experiment_json = json.load(conf_f)
        print("Deserialized JSON: {}".format(
            json.dumps(experiment_json, indent=4)))
        return Experiment(filename.split(".")[0], experiment_json, debug)
