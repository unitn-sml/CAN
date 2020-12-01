You need to use a version of pymzn that is very up to date, currently the version
on pip is too old, the pyzmn version you find here should work with this code

mnist_still_all.zip contains the whole mnist train dataset made into still life
mnist_still_dataset_10000.zip contains 1000 samples for each classe, taking the best samples
in terms of score (score is given by a pretrained NN predicting a sample belonging to its
correct class, score = prediction)


mnist_still_dataset_20000.zip contains 2000 samples for each classe, taking the best samples
in terms of score (score is given by a pretrained NN predicting a sample belonging to its
correct class, score = prediction)

The analyze_still.py script is not commented or anything, it is just a hasty script
to check statistics on the generated mnist still life dataset. This script also builds
the mnist_still_dataset_10000 and 20000.
