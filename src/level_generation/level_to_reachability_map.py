import numpy as np
import os
import sys
import json
import multiprocessing
from PIL import Image
import shutil
import argparse
from reachability import find_reachable
from generate_random_distributions import generate_distribution

SAMPLES_PER_DISTRIBUTION = 100
START_POS = (11,0)

"""
Brief description!!!
Input: levels with 13 channels, one for each type of tiles
Output: levels with 1 channels with the probability of being solid
Output: reachability distribution with 1 channels having value 1 if tile is completely reachable, 0 otherwise
"""

def load_distribution(solid_indexes, filename):
    res = np.load(filename)
    solid = res[:, :, solid_indexes].sum(axis=-1) / res.sum(axis=-1)
    return solid


def create_reachability_distribution(distribution, jumps):
    assert len(distribution.shape) == 2, "distribution: expecting a 2D numpy array with probabilities of being solid"
    mid = []
    for _ in range(SAMPLES_PER_DISTRIBUTION):
        sample = distribution > np.random.uniform(0, 1, size=distribution.shape)
        sample = find_reachable(sample, jumps, start_pos=START_POS)
        mid.append(sample)
    res = np.stack(mid, axis=0)
    res = np.mean(res, axis=0)
    return res


# main module
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platformer", type=str, required=True,
                        help="Path to the input platformer file, containing list of moves and tiles types")
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Folder containing the distributions of the levels or an integer indicating how many distributions should be generated")
    parser.add_argument("-l", "--levels_folder", type=str, required=True,
                        help="Folder to which binary level distributions will be saved")
    parser.add_argument("-r", "--reachability_folder", type=str, required=True,
                        help="Folder to which computed reachability distributions will be saved")
    parser.add_argument("-o", "--overwrite", action='store_true',
                        help="Overwrite existing files and compute reachability already done again")

    args = parser.parse_args()

    # get path of platformer file
    platformerDescription = json.load(open(args.platformer))
    print("Using platformer from file %s" % args.platformer)

    # get path of distributions or integer indicating how many distributions should be generated
    try:
        input_choice = int(args.input)
    except ValueError:
        input_choice = args.input

    # check that --levels_folder is specified when generating distributions
    levels_folder = args.levels_folder
    print("Will save generated or binarized level distributions to %s" % levels_folder)
    # eventually clean and re-create folder
    if not os.path.exists(levels_folder):
        os.makedirs(levels_folder)
        print("Level distributions folder created at %s" % levels_folder)

    # get paths of samples
    reachability_folder = args.reachability_folder
    print("Will save computed reachability distributions to %s" % reachability_folder)
    # eventually clean and re-create folder
    if not os.path.exists(reachability_folder):
        os.makedirs(reachability_folder)
        print("Reachability distributions folder created at %s" % reachability_folder)

    def worker(list_of_files):
        print("Starting worker")
        for entry in list_of_files:
            input_file, level_distribution_file, level_distribution_reachability = entry

            level_distribution = load_distribution(
                platformerDescription['solid_indexes'],
                input_file
            ) if type(input_choice) is str else generate_distribution()
            
            # create reachability distribution map from the tile distribution map
            reachability_distribution = create_reachability_distribution(level_distribution, platformerDescription['jumps'])

            # saving both the binary map and the reachability one
            np.save(level_distribution_file, level_distribution)
            np.save(level_distribution_reachability, reachability_distribution)
        print("Worker done")

    files_lists = [
        (   
            os.path.join(input_choice, filename),
            os.path.join(levels_folder, filename),
            os.path.join(reachability_folder, filename)
        ) for filename in os.listdir(input_choice) if filename.endswith(".npy") and (
            args.overwrite or (not os.path.exists(os.path.join(levels_folder, filename)) and not os.path.exists(os.path.join(reachability_folder, filename)))
        )] if type(input_choice) is str else [(   
            None,
            os.path.join(levels_folder, 'level_' + str(x) + '.npy'),
            os.path.join(reachability_folder, 'level_' + str(x) + '.npy')
        ) for x in range(input_choice)
    ]

    def split(a, n):
        k, m = divmod(len(a), n)
        return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))
    
    processes = [multiprocessing.Process(target=worker, args=(file_list,))
        for file_list in split(files_lists, multiprocessing.cpu_count())]
    
    for p in processes:
        p.start()

    for p in processes:
        p.join()

    print("Done!")
