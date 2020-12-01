#!/usr/bin/env bash
# run from src
echo "You should run me from the 'src' directory of the repository."
experiment_name=$1
echo $experiment_name
cd out
zip -r ../$experiment_name.zip images/$experiment_name
zip -r ../$experiment_name.zip log/$experiment_name
zip -r ../$experiment_name.zip model_checkpoints/$experiment_name
zip -r ../$experiment_name.zip tensorboard/$experiment_name

