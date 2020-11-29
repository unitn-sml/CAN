import tensorflow as tf
import os
import json
import scipy
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# please include 'images' or 'distributions'
IMAGES_OR_ARRAY_DISTRIBUTIONS_FLAGS = ['images']#, 'distributions']

for attr in IMAGES_OR_ARRAY_DISTRIBUTIONS_FLAGS:
    assert attr in ['images', 'distributions'], \
        "each entry in IMAGES_OR_ARRAY_DISTRIBUTIONS_FLAGS should be one between 'images' or 'distributions'"

BACKGROUND_COLOR = '#A6ECFF'

tiles_files_names = {
    0: 'solid_unbreakable.png',
    1: 'solid_breakable.png',
    2: 'air.png',
    3: 'full_question_block.png',
    4: 'empty_question_block.png',
    5: 'enemy.png',
    6: 'pipe_top_left.png',
    7: 'pipe_top_right.png',
    8: 'pipe_down_left.png',
    9: 'pipe_down_right.png',
    10: 'coin.png',
    11: 'bomb_up.png',
    12: 'bomb_down.png'
}

AVAILABLE_LEVELS = [
    'mario-1-1.txt',
    'mario-1-2.txt',
    'mario-1-3.txt',
    'mario-2-1.txt',
    'mario-3-1.txt',
    'mario-3-3.txt',
    'mario-4-1.txt',
    'mario-4-2.txt',
    'mario-5-1.txt',
    'mario-5-3.txt',
    'mario-6-1.txt',
    'mario-6-2.txt',
    'mario-6-3.txt',
    'mario-7-1.txt',
    'mario-8-1.txt'
]


######################################
######### LOADER CLASS ###############
######################################

class Loader:

    def __init__(self, config_dict, exp_name, architectures):
        self.config_dict = config_dict
        self.exp_name = exp_name
        self.architectures = architectures
        
    def get_specific_config(self):
        res_dict = {}

        # get the game to work on
        assert "GAME_NAME" in self.config_dict
        res_dict["GAME_NAME"] = self.config_dict["GAME_NAME"]
        assert res_dict["GAME_NAME"] in ['Super Mario Bros']

        # get the dataset name
        assert "DATASET_NAME" in self.config_dict
        res_dict["DATASET_NAME"] = self.config_dict["DATASET_NAME"]
        assert res_dict["DATASET_NAME"] in ['supermariobros']

        # isize is used by the generator to choose how many CNN layers should be applied
        assert "ISIZE" in self.config_dict
        res_dict["ISIZE"] = self.config_dict["ISIZE"]
        assert isinstance(res_dict["ISIZE"], int), "ISIZE must be an integer"

        # architectures parameters 
        res_dict["NUMBER_FILTERS_GENERATOR"] = self.config_dict["NUMBER_FILTERS_GENERATOR"]
        res_dict["NUMBER_FILTERS_DISCRIMINATOR"] = self.config_dict["NUMBER_FILTERS_DISCRIMINATOR"]
        assert isinstance(res_dict["NUMBER_FILTERS_GENERATOR"], int), \
            "NUMBER_FILTERS_GENERATOR must be an integer"
        assert isinstance(res_dict["NUMBER_FILTERS_DISCRIMINATOR"], int), \
            "NUMBER_FILTERS_DISCRIMINATOR must be an integer"

        # clamp weights of the generator to avoid a 
        assert "CLAMP_HIGH" in self.config_dict, "CLAMP_HIGH should be defined"
        assert "CLAMP_LOW" in self.config_dict, "CLAMP_LOW should be defined"
        res_dict["CLAMP_HIGH"] = self.config_dict["CLAMP_HIGH"]
        res_dict["CLAMP_LOW"] = self.config_dict["CLAMP_LOW"]

        # name of the dataset (mario-1-3, mario-6-2, ...)
        # assert "SPECIFIC_LEVEL" in self.config_dict, "SPECIFIC_LEVEL must be defined"
        res_dict["SPECIFIC_LEVEL"] = self.config_dict.get("SPECIFIC_LEVEL", None)

        # None, use all levels
        if res_dict["SPECIFIC_LEVEL"] is None:
            res_dict["SPECIFIC_LEVEL"] = AVAILABLE_LEVELS
        # if is a single string, transform in list with a single element
        if isinstance(res_dict["SPECIFIC_LEVEL"], str):
            res_dict["SPECIFIC_LEVEL"] = [res_dict["SPECIFIC_LEVEL"]]
        assert isinstance(res_dict["SPECIFIC_LEVEL"], list)
        # check that all levels exist
        for level_name in res_dict["SPECIFIC_LEVEL"]:
            assert level_name in AVAILABLE_LEVELS, f"{level_name} not in {AVAILABLE_LEVELS}" 

        # PIPES experiments related stuff
        # max and min number of pipes for loss PipesNumberLoss and other params
        res_dict["PIPES_NUMBER_MAX"] = self.config_dict.get("PIPES_NUMBER_MAX", 10)
        res_dict["PIPES_NUMBER_MIN"] = self.config_dict.get("PIPES_NUMBER_MIN", 10)
        assert isinstance(res_dict["PIPES_NUMBER_MAX"], int), "PIPES_NUMBER_MAX must be integer"
        assert isinstance(res_dict["PIPES_NUMBER_MIN"], int), "PIPES_NUMBER_MIN must be integer"

        # pipes number loss paramters
        res_dict["PIPES_NUMBER_LOSS_FROM_EPOCH"] = self.config_dict.get("PIPES_NUMBER_LOSS_FROM_EPOCH", 0)
        res_dict["PIPES_NUMBER_LOSS_INCREMENTAL"] = self.config_dict.get("PIPES_NUMBER_LOSS_INCREMENTAL", False)
        assert isinstance(res_dict["PIPES_NUMBER_LOSS_FROM_EPOCH"], int)
        assert isinstance(res_dict["PIPES_NUMBER_LOSS_INCREMENTAL"], bool)

        # CANNONS experiments related stuff
        # max and min number of cannons for loss CannonsNumberLoss and other params
        res_dict["CANNONS_NUMBER_MAX"] = self.config_dict.get("CANNONS_NUMBER_MAX", 10)
        res_dict["CANNONS_NUMBER_MIN"] = self.config_dict.get("CANNONS_NUMBER_MIN", 10)
        assert isinstance(res_dict["CANNONS_NUMBER_MAX"], int), "CANNONS_NUMBER_MAX must be integer"
        assert isinstance(res_dict["CANNONS_NUMBER_MIN"], int), "CANNONS_NUMBER_MIN must be integer"

        # pipes number loss paramters
        res_dict["CANNONS_NUMBER_LOSS_FROM_EPOCH"] = self.config_dict.get("CANNONS_NUMBER_LOSS_FROM_EPOCH", 0)
        res_dict["CANNONS_NUMBER_LOSS_INCREMENTAL"] = self.config_dict.get("CANNONS_NUMBER_LOSS_INCREMENTAL", False)
        assert isinstance(res_dict["CANNONS_NUMBER_LOSS_FROM_EPOCH"], int)
        assert isinstance(res_dict["CANNONS_NUMBER_LOSS_INCREMENTAL"], bool)

        # beta used for sampling. see the DCGAN_G for more info
        res_dict["BETA_SAMPLE"] = self.config_dict.get("BETA_SAMPLE", 1)
        assert isinstance(res_dict["BETA_SAMPLE"], int) or isinstance(res_dict["BETA_SAMPLE"], float), \
            "BETA_SAMPLE must be an integer or a float"

        # each branch of the if-else chain should set res_dict["TILES_MAP"]
        if self.config_dict["GAME_NAME"] == "Super Mario Bros":
            self.base_path = os.path.join("in", "datasets", "level_generation", "Super_Mario_Bros")
            self.base_path_tiles = os.path.join(self.base_path, "Tiles")
            self.base_path_reachability_models = os.path.join(self.base_path, "Models")
            self.config_path = os.path.join(self.base_path, 'smb.json')
            self.settings_path = os.path.join(self.base_path, 'mario_settings.json')
            self.data_path = os.path.join(self.base_path, 'Processed')
            # create normal and inverse tilemap
            self.tiles_map = json.load(open(self.config_path, 'r'))['tiles']
            self.settings = json.load(open(self.settings_path, 'r'))
            res_dict["SETTINGS"] = self.settings
            self.inverse_tiles_map = dict()
            for index, key in enumerate(self.tiles_map):
                tmp = self.tiles_map[key]
                self.tiles_map[key] = {
                    'index': index,
                    'value': tmp
                }
                self.inverse_tiles_map[index] = {
                    'key': key,
                    'value': tmp
                }
            res_dict["TILES_MAP"] = self.tiles_map
            res_dict["INVERSE_TILE_MAP"] = self.inverse_tiles_map
            self.tiles_files = {key: Image.open(os.path.join(self.base_path_tiles, value)) for (key, value) in tiles_files_names.items()}

            if "REACHABILITY_PRETRAINED_MODEL" in self.config_dict:
                res_dict["REACHABILITY_PRETRAINED_MODEL"] = os.path.join(
                    self.base_path_reachability_models, self.config_dict.get("REACHABILITY_PRETRAINED_MODEL"))
            
            if "REACHABILITY_ARCHITECTURE" in self.config_dict:
                res_dict["REACHABILITY_ARCHITECTURE"] = self.architectures[self.config_dict.get("REACHABILITY_ARCHITECTURE")]
            
            assert self.config_dict["SHAPE"][2] == len(self.tiles_map), \
                "experiment declares a wrong shape[2] (# channels) : {} vs {}".format(self.config_dict["SHAPE"][2], len(self.tiles_map))
        else:
            raise NotImplementedError("Training on game {} not implemented yet".format(self.config_dict["GAME_NAME"]))
        return res_dict


    def plot_data(self, samples, epoch, evaluation=False):
        # I'm a comment
        base_path = os.path.join("out", "images", self.exp_name)
        # distribution paths
        distributions_path = os.path.join(base_path, "evaluation-distributions" if evaluation else "training-distributions")
        name_distributions = "evaluation-distribution-{}.npy" if evaluation else "distribution-epoch-%d-number-{}.npy" % epoch

        # images path
        images_path = os.path.join(base_path, "evaluation-images" if evaluation else "training-images")
        name_images = "evaluation-images-{}.png" if evaluation else "sample-epoch-%d-number-{}.png" % epoch

        # save distributions as numpy files, ready to be used with np.load()
        if 'distributions' in IMAGES_OR_ARRAY_DISTRIBUTIONS_FLAGS or evaluation:
            # samples shape: (eval_samples, width, height, channels). it is a numpy array.
            # check if folder already exists and eventually create it
            if not os.path.exists(distributions_path):
                os.mkdir(distributions_path)
            for j, sample in enumerate(samples):
                path = os.path.join(distributions_path, name_distributions.format(j))
                np.save(path, sample)

        # save images as PNG mario files
        if 'images' in IMAGES_OR_ARRAY_DISTRIBUTIONS_FLAGS or evaluation:
            one_hot_samples = self._one_hot_rev(samples)
            # check if folder already exists and eventually create it
            if not os.path.exists(images_path):
                os.mkdir(images_path)
            for j, sample in enumerate(one_hot_samples):
                res = Image.new('RGBA', (sample.shape[1]*16, sample.shape[0]*16), BACKGROUND_COLOR)
                for ii in range(sample.shape[0]):
                    for jj in range(sample.shape[1]):
                        res.paste(self.tiles_files[sample[ii][jj]], (jj*16, ii*16))
                res.save(os.path.join(images_path, name_images.format(j)), format='png')


    def get_dataset(self):
        if self.config_dict['GAME_NAME'] == 'Super Mario Bros':
            level_list = self._get_super_mario_bros()
            #print(len(level_list))
            #exit()
            self.plot_data(np.asarray(level_list), -1)
            train_len, test_len, validation_len = self.config_dict["TRAINING_TEST_VALIDATION_SPLITS"]
            np.random.shuffle(level_list)
            # if splits are expressed as floats, convert them in integers
            if isinstance(train_len, float):
                train_len = round(train_len * len(level_list))
                test_len = round(test_len * len(level_list))
                validation_len = len(level_list) - test_len - train_len
            assert train_len + test_len + validation_len == len(level_list), \
                "TRAINING_TEST_VALIDATION_SPLITS values do not sum to the total lenght of the dataset ({0})".format(len(level_list))
            return np.split(level_list, [train_len, train_len + test_len])
        else:
            raise NotImplementedError(self.config_dict['GAME_NAME'] + ' not yet implemented')


    ################################################
    ##### LOADERS OF DIFFERENT GAME'S DATASETS #####
    ################################################
    def _one_hot(self, targets):
        '''
        convert numpy level to one hot encoding
        '''
        flattened = targets.reshape(-1)
        nb_classes = self.config_dict["SHAPE"][-1]
        res = np.eye(nb_classes)[flattened]
        res = res.reshape(list(targets.shape) + [nb_classes])
        return res

    def _one_hot_rev(self, targets):
        '''
        convert numpy levels from one-hot encoding to classic encoding
        '''
        return np.argmax(targets, axis=-1)

    # SUPER MARIO
    def _txt2array(self, txt_filename):
        with open(txt_filename, 'r') as f:
            res = []
            for line in f:
                row = [self.tiles_map[char]['index'] for char in line.strip()]
                if len(row) > 0:
                    res.append(row)
        res = np.array(res)
        return res

    def _array2txt(self, txt_filename, array):
        with open(txt_filename, 'w') as f:
            for line in array:
                row = "".join([self.inverse_tiles_map[number]['key']
                               for number in line])
                if len(row) > 0:
                    f.write(row + "\n")

    def _sample_super_mario_level(self, level, shape):
        total_height = level.shape[0]
        total_width = level.shape[1]
        sample_height = shape[0]
        sample_width = shape[1]
        assert total_height == sample_height
        res = np.zeros((total_width - sample_width + 1, sample_height, sample_width), dtype=int)
        for i in range(0, total_width - sample_width + 1):
            res[i, :sample_height, :sample_width] = level[:, i:i+sample_width]
        return res

    # Super Mario Bros
    def _get_super_mario_bros(self):
        level_shape = self.config_dict["SHAPE"][:2]
        # Read files in 'tile-txt' format
        level_list = []
        level_names = self.config_dict.get("SPECIFIC_LEVEL")
        for filename in os.listdir(self.data_path):
            if filename in level_names:
                filepath = os.path.join(self.data_path, filename)
                level_list.append(self._sample_super_mario_level(self._txt2array(filepath), level_shape))

        data = np.concatenate(level_list)
        data = self._one_hot(data)
        return data
