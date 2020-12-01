#!/bin/bash

# Run some experiments on cannons consistency.
# To have stable results, you should run all the cannons experiments from the in/experiment/level_generation/test folder.
# If you have enough GPU memory, you may want to run them in parallel. With 11GB we were able to launch 4 experiments at a time.

cd ..
for level in in/experiments/level_generation/mario-7-1/cannons/*; do
    python main.py -i $level -f 0.5
done