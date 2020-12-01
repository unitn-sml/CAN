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



def compute_constraint_satisfaction(image, threshold):
    img_width = image[1].shape[0]
    data = image[1]
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
    return satisfaction, (data if satisfaction >= threshold else None)

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



def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')



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
    parser.add_argument("--path_can",
                        help="image path gan_sl", type=str,
                        required=True, default="")
    parser.add_argument("--path_gan",
                        help="image path gan", type=str,
                        required=True, default="")
    parser.add_argument("--threshold", help="validity threshold", type=int,
                        required=False, default=100)


    args = parser.parse_args()
    exp_data_can = get_experiment_images(args.path_can)
    exp_data_gan = get_experiment_images(args.path_gan)
    perfects_can = []
    perfects_gan = []
    for exp, data in exp_data_can:
        for image in data:
            satisfaction, data = compute_constraint_satisfaction(image, args.threshold)
            if data is not None:
                perfects_can.append(data)
    for exp, data in exp_data_gan:
        for image in data:
            satisfaction, data = compute_constraint_satisfaction(image, args.threshold)
            if data is not None:
                perfects_gan.append(data)
    perfects_can_reshaped = np.reshape(np.array(perfects_can), (np.array(perfects_can).shape[0], -1))
    perfects_gan_reshaped = np.reshape(np.array(perfects_gan), (np.array(perfects_gan).shape[0], -1))
    unique_can = np.unique(perfects_can_reshaped, axis=0)
    unique_gan = np.unique(perfects_gan_reshaped, axis=0)

    labels=["Valid", "Unique"]
    gan_values = [perfects_gan_reshaped.shape[0], unique_gan.shape[0]]
    can_values = [perfects_can_reshaped.shape[0], unique_can.shape[0]]
    dpi = 300
    plt.figure(figsize=(16, 9), dpi=dpi)
    fig, ax = plt.subplots()
    x_labels = np.arange(len(labels))
    rects1 = ax.bar(x_labels - 0.35 / 2, can_values, 0.35, label=r"CAN $\lambda$ = 1", color="C1")
    rects2 = ax.bar(x_labels + 0.35 / 2, gan_values, 0.35,
                    label='GAN', color="C2")
    ax.set_ylabel('Count')
    ax.set_xticks(x_labels)
    ax.set_xticklabels(labels)
    ax.legend()
    autolabel(rects1)
    autolabel(rects2)
    fig.tight_layout()
    plt.savefig(f"Uniqueness_GAN_CAN_{args.threshold}.png",  bbox_inches="tight")
