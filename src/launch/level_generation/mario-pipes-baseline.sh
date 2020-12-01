# Run the four experiments of GAN using different seeds
# Notice that this scripts are equivalent to a GAN but are CANs with the semantic loss weight lambda set to 0
# This allows Validity, Novelty and Uniqueness to be measured. For this reason run times are similar
# to CANs and not lower. 

PREFIX=in/experiments/level_generation/test
for f in ${PREFIX}/mario-7-pipes-sl-base-test-*;
do
    python main.py -i $f
done
