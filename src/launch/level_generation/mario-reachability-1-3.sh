# Run the four experiments of CAN on the reachability constraint using different seeds
# Reachability is an approximation of the A* agent performance: to know the exact 
# validity of reachability, see https://github.com/TheHedgeify/DagstuhlGAN

PREFIX=in/experiments/level_generation/test
for f in ${PREFIX}/mario-1-3-reachability-001-5000-test-*;
do
    python main.py -i $f
done
