import os
import json
import numpy as np
import argparse
from PIL import Image

######################################
######### LOADER CLASS ###############
######################################
BACKGROUND_COLOR = 'white'


"""
# to be used implementing binary -> image
def save_image_from_level(path, level):
    assert len(level.shape) == 2
    img = Image.fromarray((level * 255).astype('uint8'), mode='L')
    img.save(path)
"""

class Converter:
    """
    Convert levels from some format to others...
    """
    def __init__(self):
        self.base_path = os.path.join("../..", "in", "datasets", "level_generation", "Super_Mario_Bros")
        self.settings = json.load(
            open(os.path.join(self.base_path, "mario_settings.json"), 'r'))
        self.tiles_map = self.settings['tiles']
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
        self.base_path_tiles = os.path.join(self.base_path, "Tiles")
        self.tiles_files = {key: Image.open(os.path.join(
            self.base_path_tiles, value)) for (key, value) in self.settings['tiles_files_names'].items()}

    def _load_txt_dataset_from_folder(self, folder):
        """
        Keep al files in folder that ends with txt and return them as arrays
        :return: a list of numpy arrays in one-hot encoding and their names
        """
        res = []
        names = []
        for filename in os.listdir(folder):
            filename_path = os.fsdecode(filename)
            if filename.endswith(".txt"):
                sample = self._txt2array(os.path.join(folder, filename_path))
                sample = self._one_hot(sample)
                names.append(filename)
                res.append(sample)
        return names, res

    def _load_numpy_dataset_from_folder(self, folder):
        """
        Keep al files in folder that ends with txt and return them as arrays
        :return: a list of numpy arrays in one-hot encoding and their names
        """
        res = []
        names = []
        for filename in os.listdir(folder):
            filename_path = os.fsdecode(filename)
            if filename.endswith(".npy"):
                sample = np.load(os.path.join(folder, filename_path))
                names.append(filename)
                res.append(sample)
        # list to numpy array
        res = np.asarray(res)
        return names, res

    def _load_txt_dataset_from_file(self, filename, height=14, width=28):
        """
        Read a file and split it in fixed height and width levels
        :return: a list of arrays in one-hot encoding
        """
        # Read files in 'tile-txt' format
        data = self._split_super_mario_level(self._txt2array(filename), sample_height=height, sample_width=width)
        # convert to one-hot encoding
        data = self._one_hot(data)
        return data

    def _save_dataset_as_numpy(self, levels, names, folder):
        """
        Save all levels in levels as numpy arrays with names given by names in the output folder folder.
        :return: None
        """
        for c in zip(levels, names):
            level, name = c
            np.save(os.path.join(folder, name), level)

    def _save_dataset_as_json(self, levels, folder):
        """
        Save all levels in a single json file with name folder.
        :return: None
        """
        if len(levels.shape) == 4:
            levels = self._one_hot_rev(levels)
        
        levels = levels.astype(int).tolist()
        res = json.dumps(levels)
        with open(folder, "w") as f:
            f.write(res)

    def _save_dataset_as_img(self, levels, names, folder):
        """
        Save all levels in levels as pictures with names given by names in the output folder folder.
        :return: None
        """
        for c in zip(levels, names):
            level, name = c
            if len(level.shape) == 3:
                level = self._one_hot_rev(level)
            res = Image.new('RGBA', (level.shape[1]*16, level.shape[0]*16), BACKGROUND_COLOR)
            for ii in range(level.shape[0]):
                for jj in range(level.shape[1]):
                    res.paste(self.tiles_files[str(level[ii][jj])], (jj*16, ii*16))
                res.save(os.path.join(folder, name), format='png')

    def convert(self, input_folder, input_type, output_folder, output_type):
        """
        main conversion function, calls the required specific 
        converted based on the input and output types
        :return: None
        """
        if input_type == "txt" and output_type == "numpy":
            self._txt2npy(input_folder, output_folder)
        elif input_type == "numpy" and output_type == "img":
            self._npy2img(input_folder, output_folder)
        elif input_type == "numpy" and output_type == "json":
            self._npy2json(input_folder, output_folder)
        else:
            raise NotImplementedError("This conversion type is not yet implemented")
    
    def _npy2json(self, input_folder, output_folder):
        """
        Convert all the numpy files in a folder input_folder and save them as single json file. 
        If input_folder is a single file then it is split in different levels of fixed width.
        If output_folder does exist, it is emptied.
        :return: None
        """
        # in case input_folder is really a folder
        if os.path.isdir(input_folder):
            _, levels = self._load_numpy_dataset_from_folder(input_folder)

        # if output folder does already exist, delete and re-create it
        if os.path.exists(output_folder):
            os.remove(output_folder)

        self._save_dataset_as_json(levels, output_folder)


    def _npy2img(self, input_folder, output_folder):
        """
        Convert all the numpy files in a folder input_folder and save them as numpy arrays in the output_folder. 
        If input_folder is a single file then it is split in different levels of fixed width.
        If output_folder does exist, it is emptied.
        :return: None
        """
        # in case input_folder is really a folder
        if os.path.isdir(input_folder):
            names, levels = self._load_numpy_dataset_from_folder(input_folder)
            names = ["{}.png".format(x.split('.')[0]) for x in names]

        # if output folder does already exist, delete and re-create it
        if os.path.exists(output_folder):
            import shutil
            shutil.rmtree(output_folder)
        os.mkdir(output_folder)

        self._save_dataset_as_img(levels, names, output_folder)

    def _txt2npy(self, input_folder, output_folder):
        """
        Convert all the TXT files in a folder input_folder and save them as numpy arrays in the output_folder. 
        If input_folder is a single file then it is split in different levels of fixed width.
        If output_folder does exist, it is emptied.
        :return: None
        """
        # in case input_folder is really a folder
        if os.path.isdir(input_folder):
            names, levels = self._load_txt_dataset_from_folder(input_folder)
            names = ["{}.npy".format(x.split('.')[0]) for x in names]
        # input_folder is a single file that has to be splitted
        else:
            levels = self._load_txt_dataset_from_file(input_folder)
            base_name = input_folder.split('/')[-1].split('.')[0]
            names = ["{}-{}.npy".format(base_name, i) for i in range(len(levels))]
        if os.path.exists(output_folder):
            import shutil
            shutil.rmtree(output_folder)
        os.mkdir(output_folder)
        
        self._save_dataset_as_numpy(levels, names, output_folder)

    def _one_hot(self, targets):
        '''
        convert numpy level to one hot encoding
        '''
        flattened = targets.reshape(-1)
        nb_classes = len(self.tiles_map)
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
        """
        Return the numpy array given a txt filename
        """
        res = []
        with open(txt_filename, 'r') as f:
            for line in f:
                row = [self.tiles_map[char]['index'] for char in line.strip()]
                if len(row) > 0:
                    res.append(row)
        res = np.array(res)
        return res

    def _split_super_mario_level(self, level, sample_height=14, sample_width=28):
        """
        Split a super mario bros level on the width in levels with a fixed width
        """
        total_height = level.shape[0]
        total_width = level.shape[1]
        assert total_height == sample_height
        res = np.zeros((total_width - sample_width + 1, sample_height, sample_width), dtype=int)
        for i in range(0, total_width - sample_width + 1):
            res[i, :, :] = level[:, i:i+sample_width]
        return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Path to the input textfile samples")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Path to the output numpy samples")
    parser.add_argument("--input_type", type=str, required=True,
                        help="Input file tipe, one of 'numpy' or 'txt'")
    parser.add_argument("--output_type", type=str, required=True,
                    help="Output file tipe, one of 'numpy', 'txt' or 'img'")

    args = parser.parse_args()
    # i/o folders
    input_folder = args.input
    output_folder = args.output
    input_type = args.input_type
    output_type = args.output_type

    if not os.path.exists(input_folder):
        print("Should specify correct input path")

    print("Starting conversion")
    converter = Converter()
    converter.convert(input_folder, input_type, output_folder, output_type)
    print("Done!")
    exit()

