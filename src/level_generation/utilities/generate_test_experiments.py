"""
Given an experiment, generate a sequence of equal experiments with different seeds and save them in the output folder
"""
import argparse
import os
import json
import random

def main(experiment_paths, output_folder, number):
    # check json experiment exists
    for experiment_path in experiment_paths:
        assert os.path.exists(experiment_path), "File {} does not exists".format(experiment_path)

    if not os.path.exists(output_folder):
        print("Creating output folder {}".format(output_folder))
        os.mkdir(output_folder)

    print("Opening experiment json")
    try:
        experiments = [json.loads(open(experiment_path, 'r').read()) for experiment_path in experiment_paths]
    except json.JSONDecodeError:
        print("Failed to decode an experiment json file, exiting")
        exit(1)

    experiment_names = [os.path.split(experiment_path)[1] for experiment_path in experiment_paths]

    for c in zip(experiment_names, experiments):
        experiment_name, experiment = c
        
        for i in range(number):
            experiment["ANN_SEED"] = random.randint(1000, 100000)
            name = "{}-test-{}.json".format(
                ".".join(experiment_name.split('.')[:-1]),
                i
            )
            serialized_json = json.dumps(experiment)
            filepath = os.path.join(output_folder, name)

            with open(filepath, "w") as f:
                f.write(serialized_json)

    print("Done!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment", required=True, type=str, nargs='+',
                        help="Path to the experiment json file from which start the generation")
    parser.add_argument("-o", "--output_folder", required=True, type=str,
                        help="The folder to which save the experiments")
    parser.add_argument("-n", "--number", required=True, type=int,
                        help="The max y value on the plot")
    args = parser.parse_args()

    main(args.experiment, args.output_folder, args.number)
