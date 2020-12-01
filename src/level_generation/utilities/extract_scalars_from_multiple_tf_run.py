from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import argparse
import os
import re
import tensorflow

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_folder", type=str, required=True,
                    help="Path to the folder containing multiple tensorboard exps")
parser.add_argument("--scalars", type=str, required=True, nargs="+",
                    help="Name of scalars to be extracted")
parser.add_argument("--group", type=str, required=False, default="-test-[0-9]",
                    help="Name of scalars to be extracted")
parser.add_argument("--step_interval", type=int, required=False, default=[11799, 11999], nargs=2,
                    help="Average value on last n steps")

args = parser.parse_args()

global_map = dict()
# this will contain thing like
"""
{
    "mario-6-2-001-0001-5000": [event_1, event_2],
    ...
}
"""

assert os.path.isdir(args.input_folder), f"{args.input_folder} must be a valid path to a folder"
args.scalars = [x.lower() for x in args.scalars]

for directory in os.listdir(args.input_folder):
    directory = os.path.join(args.input_folder, directory)
    if os.path.isdir(directory):
        name = re.sub(args.group, '', os.path.basename(directory))
        event_file = os.listdir(directory)[0]
        assert event_file.startswith("events.out.tfevents."), f"File {event_file} in dir {directory} does not start with 'events.out.tfevents.'"
        
        if name not in global_map:
            global_map[name] = []
        global_map[name].append(
            os.path.join(args.input_folder, directory, event_file)
        )


scalar_map = dict()

for name, event_files in global_map.items():
    
    scalar_map[name] = dict()

    for event_file in event_files:
        #print(name, event_file)
        # Loading too much data is slow...
        tf_size_guidance = {
            'compressedHistograms': 10,
            'images': 0,
            'scalars': 100,
            'histograms': 1
        }

        event_acc = EventAccumulator(event_file, tf_size_guidance)
        event_acc.Reload()

        # Show all tags in the log file
        
        for scalar in args.scalars:
            for tb_scalar in event_acc.Tags()["scalars"]:
                #print(f"Compare '{tb_scalar}' and '{scalar}'")
                if tb_scalar.lower().endswith(scalar):

                    if scalar not in scalar_map[name]:
                        scalar_map[name][scalar] = []

                    scalar_map[name][scalar] += [x.value for x in event_acc.Scalars(tb_scalar) if (x.step >= args.step_interval[0] and x.step <= args.step_interval[1])]

#import json
#print(json.dumps(scalar_map, indent=4))

for name, data in scalar_map.items():
    for metric, value in data.items():
        print(f"{name} {metric}: {sum(value)/len(value)}")