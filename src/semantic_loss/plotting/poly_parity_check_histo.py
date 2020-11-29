import matplotlib.pyplot as plt
import scipy.misc
import scipy.ndimage
import numpy as np
import os
from os.path import isfile, join
import argparse
from numpy import sum as np_sum
from matplotlib import pyplot as plt
from collections import Counter

"""
Simple script to check the directory 'images' for experiment results directories to compare the last epoch (250) 
generated evaluation batch of different experiments.
"""

def compute_constraint_satisfaction(image):
    #print(image[0])
    img_width = image[1].shape[0]
    data = image[1]
    #print(data.shape)
    sum_rows_l = np_sum(data[1:(img_width - 1), 1:int(img_width / 2)],
                        1)
    check_rows_l = np.mod(sum_rows_l, 2)
    check_rows_l = np.equal(data[1:(img_width - 1),0],
                                check_rows_l).astype(float)

    sum_rows_r = np_sum(
        data[1:(img_width - 1), int(img_width / 2):(img_width - 1)], 1)
    check_rows_r = np.mod(sum_rows_r, 2)
    check_rows_r = np.equal(data[1:(img_width - 1), img_width - 1],
                                check_rows_r).astype(float)
    satisfaction = np.mean(np.concatenate([check_rows_l, check_rows_r]))*100
    return satisfaction, (data if satisfaction == 100.0 else None)

def get_last_batch(directory):
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            abspath = os.path.abspath(join(dirpath, f))
            if isfile(abspath) and abspath[-3:] == "png" and "251" in abspath:
                img = scipy.ndimage.imread(abspath, flatten=True)
                img[img <= 127] = 0
                img[img > 127] = 1
                img = img.astype(np.int32)
                assert np.max(img) == 1 or np.max(img) == 0
                assert np.min(img) == 0
                values = len(set(list(np.reshape(img, -1))))
                assert values == 2 or values == 1
                yield (abspath.split("/")[-1], img)


def get_experiment_images(path="images"):
    data = []
    for root, dirs, files in os.walk(path):
        fullpath = os.path.abspath(root)
        images = list(get_last_batch(fullpath))
        images = sorted(images, key=lambda x: x[0])
        data.append((dir, images))
    return data




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path",
                        help="image path", type=str,
                        required=False, default="")
    args = parser.parse_args()
    exp_data = get_experiment_images(args.path)
    constraints_safisfaction = []
    perfects = []
    for exp, data in exp_data:
        for image in data:
            satisfaction, data = compute_constraint_satisfaction(image)
            constraints_safisfaction.append(satisfaction)
            if data is not None:
                perfects.append(data)
    #print(Counter(constraints_safisfaction))
    print(np.array(perfects).shape)
    print(np.unique(np.reshape(np.array(perfects), (np.array(perfects).shape[0], -1)), axis=0).shape)
    # fixed bin size
    bins = np.arange(0, 100, 10)  # fixed bin size
    plt.hist(constraints_safisfaction, bins=np.arange(60,111,10), alpha=0.5)
    plt.title('percentage constraints satisfaction SL = 0.5')
    plt.xlabel('variable X (bin size = 5)')
    plt.xticks(np.arange(60,111,10))
    plt.yticks(np.arange(0,301,20))
    plt.ylabel('count')
    plt.show()
    #plt.savefig("SL05.png")
