# Run the four experiments of CAN on the pipes constraint using different seeds

PREFIX=in/experiments/level_generation/test
for f in ${PREFIX}/mario-7-pipes-sl-02-5000-test-*;
do
    python main.py -i $f
done
