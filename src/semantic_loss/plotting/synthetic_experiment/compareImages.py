import matplotlib.pyplot as plt
import scipy.misc
import scipy.ndimage
import numpy as np
import os
from os.path import isfile, join

"""
Simple script to check the directory 'images' for experiment results directories to compare the last epoch (250) 
generated evaluation batch of different experiments.
"""


def get_last_batch(directory):
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            abspath = os.path.abspath(join(dirpath, f))
            if isfile(abspath) and abspath[-3:] == "png" and "250" in abspath:
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
        for dir in dirs:
            fullpath = os.path.abspath(join(root, dir))
            images = list(get_last_batch(fullpath))
            images = sorted(images, key=lambda x: x[0])
            data.append((dir, images))
    return data


fig = plt.figure(figsize=(20, 20))

title = "Sample pictures from the last generated batch"
fig.suptitle(title)
i = 1

exp_data = get_experiment_images()
exp_data = sorted(exp_data, key=lambda x: x[0])
nexperiments = len(exp_data)
nimages = len(exp_data[1][1])
for exp, data in exp_data:
    for img in data:
        ax1 = fig.add_subplot(nexperiments, nimages, i)
        if (i - 1) % len(data) == 0:
            plt.title(exp)
        plt.imshow(img[1])
        plt.axis("off")
        i += 1
plt.subplots_adjust(wspace=0., hspace=0.4)
plt.show()
