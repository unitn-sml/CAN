"""
Given an experiment, generate a new experiment with a custom width, height and seed and it seves the new experiment in the same folder of the original one
"""
import argparse
import os
import json
import random

def main(experiment_paths, img_width, img_height, seed):
    # check json experiment exists
    for experiment_path in experiment_paths:
        assert os.path.exists(experiment_path), "File {} does not exists".format(experiment_path)
    print("Opening experiment json")
    try:
        experiments = [json.loads(open(experiment_path, 'r').read()) for experiment_path in experiment_paths]
    except json.JSONDecodeError:
        print("Failed to decode an experiment json file, exiting")
        exit(1)
    experiment_names = [os.path.split(experiment_path)[1] for experiment_path in experiment_paths]
    experiment_folders = [os.path.split(experiment_path)[0] for experiment_path in experiment_paths]
    for c in zip(experiment_names, experiments, experiment_folders):
        experiment_name, experiment, experiment_folder = c
        experiment["ANN_SEED"] = seed
        experiment["IMG_WIDTH"] = img_width
        experiment["IMG_HEIGHT"] = img_height
        if "run" not in experiment_name:
            name = "{}_run{}_height{}_width{}.json".format(
                ".".join(experiment_name.split('.')[:-1]),
                seed, img_height, img_width
            )
        else:
            name = "{}_run{}_height{}_width{}.json".format(
                ".".join(experiment_name.split('.')[:-1])[:-5],
                seed, img_height, img_width
            )
        filepath = os.path.join(experiment_folder, name)
        with open(filepath, "w") as f:
            json.dump(experiment, f, indent=4)

    print("Done!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment", required=True, type=str, nargs='+',
                        help="Path to the experiment json file from which start the generation")
    parser.add_argument("-w", "--img_width", required=True, type=int,
                        help="Image width")
    parser.add_argument("-H", "--img_height", required=True, type=int,
                        help="Image height")
    parser.add_argument("-s", "--seed", required=True, type=int,
                        help="seed")

    args = parser.parse_args()

    main(args.experiment, args.img_width,
         args.img_height, args.seed)