import pymzn
import argparse
from pymzn import chuffed
import scipy.misc
import scipy.ndimage
import numpy as np
import os
from os.path import isfile, join
from multiprocessing import Pool


def read_binarized(files_names):
    images_data = []
    for image in files_names:
        # read image as greyscale
        image_data = scipy.ndimage.imread(image, flatten=True)
        assert np.max(image_data) <= 255
        assert np.min(image_data) >= 0
        image_data[image_data <= 127] = 0  # binarize (255 -> 1)
        image_data[image_data > 127] = 1  # binarize (255 -> 1)
        image_data = image_data.astype(np.int32)
        images_data.append(image_data)
    return images_data


def get_images(directory):
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            abspath = os.path.abspath(join(dirpath, f))
            if isfile(abspath):
                yield abspath


def look_for_solutions(data):
    # binary search
    solns = []
    left = 0
    right = np.sum(data["input"])
    timeout = 1

    while left <= right:
        index = (right + left) // 2
        if index == 0:
            # make sure it has time to find the all zeros solution
            timeout = 1000

        data["min_acceptable"] = index
        tmpsols = pymzn.minizinc("still_life.mzn", "still_life.dzn", data=data, num_solutions="1", parallel="1",
                                 solver=chuffed, seed="1337", timeout=timeout)
        if len(tmpsols) == 0:
            right = index - 1
        else:
            left = index + 1
            solns = tmpsols

    return solns


def _my_job(imagenames, imgs):
    if len(imgs) > 0:
        n = imgs[0].shape[0]
        assert n == imgs[0].shape[1]
        data = dict()
        data["n"] = n

        for name, image in zip(imagenames, imgs):
            data["input"] = image

            solns = look_for_solutions(data=data)

            scipy.misc.imsave("mnist_train_still/" + name.split("/")[-1], solns[0]["a"])


# stackoverflow
def chunks(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def process_data(imagenames, imgs, nprocesses):
    # open pool of processes, each one will contribute to parsing part of the constraints

    names_chunks = chunks(imagenames, nprocesses)
    imgs_chunks = chunks(imgs, nprocesses)

    with Pool(processes=nprocesses) as pool:
        jobs = []
        for nchunk, ichunk in zip(names_chunks, imgs_chunks):
            jobs.append(pool.apply_async(_my_job, [nchunk, ichunk]))

        [job.get() for job in jobs]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--minizinc", type=str, required=True, help="Path to minizinc binary")
    parser.add_argument("-p", "--processes", type=int, required=False, default=1, help="Number of parallel processes")
    parser.add_argument("-t", "--testing", required=False, default=False, action="store_true",
                        help="If you are just testing the run time of an optimizer, limit at 1 image per process")
    args = parser.parse_args()
    pymzn.config['minizinc'] = args.minizinc

    imgnames = list(get_images("mnist_train"))
    if args.testing:
        imgnames = imgnames[:args.processes]

    # just work on stuff that has not already been done
    imgnames = [name for name in imgnames if (not os.path.isfile("mnist_train_still/" + name.split("/")[-1]))]
    images = read_binarized(imgnames)

    os.makedirs("mnist_train_still", exist_ok=True)

    process_data(imgnames, images, args.processes)

    # to make sure all images have been processed/have a still life counterpart
    assert all([os.path.isfile(f) for f in ["mnist_train_still/" + name.split("/")[-1] for name in imgnames]])
