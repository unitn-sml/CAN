#!/usr/bin/env bash
# run from src
experiment_name=$1
echo $experiment_name
bgans_path="../out/images/$experiment_name"
echo $bgans_path
checkpoints_path="../out/model_checkpoints/$experiment_name"
echo $checkpoints_path
tensorboard_path="../out/tensorboard/$experiment_name"
echo $tensorboard_path
log_path="../out/log/$experiment_name"
echo $log_path
rm -r $bgans_path*
rm -r $checkpoints_path*
rm -r $tensorboard_path*
rm -r $log_path
