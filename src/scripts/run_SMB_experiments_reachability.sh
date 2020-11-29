#!/bin/bash

# Run some experiments on reachability.
# To have stable results, you should run all the reachability experiments from the in/experiment/level_generation/test folder.
# If you have enough GPU memory, you may want to run them in parallel. With 11GB we were able to launch 4 experiments at a time.

# Choose the level between mario-1-3 and mario-3-3
LEVEL="mario-1-3"
# Choose the constraint to apply: "reachability" = normal playability of levels; "reachability_astar" = optimized to score well with A*
CONSTRAINT="reachability"

cd ..
for level in in/experiments/level_generation/${LEVEL}/${CONSTRAINT}/*; do
    python main.py -i $level -f 0.5
done