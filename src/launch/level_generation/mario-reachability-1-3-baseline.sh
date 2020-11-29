# Run the four experiments of CAN on the reachability constraint using different seeds
# Notice that this script are equivalent to a GAN but are CANs with the semantic loss weight lambda set to 0
# This allows Validity, Novelty, Uniqueness and Reachability to be measured. For this reason run times are similar
# to CANs and not lower. 
# Reachability is an approximation of the A* agent performance: to know the exact 
# validity of reachability, see https://github.com/TheHedgeify/DagstuhlGAN

PREFIX=in/experiments/level_generation/test
for f in ${PREFIX}/mario-1-3-reachability-base-test-*;
do
    python main.py -i $f
done
