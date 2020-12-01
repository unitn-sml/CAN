import numpy as np
import os
import sys
import json
import multiprocessing
import shutil
import argparse


def generate_distribution(height=14, width=28):
    distribution = np.random.uniform(low=0.0, high=1.0, size=(height, width))
    return distribution


# main module
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--nsamples", type=int, required=True,
                        help="Number of distributions")
    parser.add_argument("-l", "--levels_folder", type=str, required=True,
                        help="Folder to which binary distributions will be saved")

    args = parser.parse_args()

    # check that --levels_folder is specified when generating distributions
    levels_folder = args.levels_folder
    print("Will save generated level distributions to %s" % levels_folder)

    if not os.path.exists(levels_folder):
        os.mkdir(levels_folder)

    # check that --levels_folder is specified when generating distributions
    n_samples = args.nsamples
    print("Will create %d samples" % n_samples)

    for i in range(n_samples):
        # saving both the binary map and the reachability one
        np.save(os.path.join(levels_folder, 'generated_distrib_{}'.format(i)), generate_distribution())

    print("Done!")

    
        
