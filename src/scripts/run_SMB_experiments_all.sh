#!/bin/bash

# Run all the test experiments.
# This may require a lot of time!
# If you have enough GPU memory, you may want to run them in parallel. With 11GB we were able to launch 4 experiments at a time.

cd ..
for level in in/experiments/level_generation/test/*; do
    python main.py -i $level -f 0.5
done