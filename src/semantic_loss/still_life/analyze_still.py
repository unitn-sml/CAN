import matplotlib.pyplot as plt
import scipy.misc
import scipy.ndimage
import numpy as np
import os
import random
import torch.nn as nn
import torch
from collections import OrderedDict
import torch.utils.model_zoo as model_zoo
from os.path import isfile, join


###################
# plot number of bits set to 1
##################################################################
##################################################################
##################################################################
##################################################################
##################################################################

def get_images(directory):
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            abspath = os.path.abspath(join(dirpath, f))
            if isfile(abspath):
                img = scipy.ndimage.imread(abspath, flatten=True)
                img[img <= 127] = 0
                img[img > 127] = 1
                img = img.astype(np.int32)
                assert np.max(img) == 1 or np.max(img) == 0
                assert np.min(img) == 0
                values = len(set(list(np.reshape(img, -1))))
                assert values == 2 or values == 1
                yield (abspath.split("/")[-1], img)


def split_by_class(data):
    res = dict()
    for i in range(10):
        res[str(i)] = []

    for name, img in data:
        digit_class = name.split(".")[0][-1]  # remove .png and get the last character
        res[digit_class].append((name, img))

    return res


# aggregation on all data
still = list(get_images("mnist_still_all"))
real = list(get_images("mnist_train"))
assert len(still) > 0
assert len(real) > 0
sumsstill = [np.sum(img) for name, img in still]
sumsreal = [np.sum(img) for name, img in real]

# plt.hist([sumsstill, sumsreal], bins=max(sumsstill + sumsreal))  # , histtype='step')
# plt.legend(["still_life", "binarized"])
# plt.xlabel('Number of 1 bits');
# plt.ylabel('Number of Images');
# plt.show()


# aggregation by classes
still_classes = split_by_class(still)
real_classes = split_by_class(real)
still_classes_sums = {k: [np.sum(img) for name, img in v] for k, v in still_classes.items()}
real_classes_sums = {k: [np.sum(img) for name, img in v] for k, v in real_classes.items()}

fig = plt.figure(figsize=(20, 20))
fig.suptitle('Histograms of the total bits set to 1 for each class, real binarized data vs still life data')

for i in range(10):
    ax1 = fig.add_subplot(2, 5, i + 1)

    classtype = str(i)
    to_plot = [still_classes_sums[classtype]] + [real_classes_sums[classtype]]
    bins = [i for i in range(301)]
    bins = sorted(bins + bins)
    n, bins, patches = plt.hist(to_plot[::-1], bins=bins, histtype='stepfilled', alpha=0.8)
    plt.legend(["mnist_still_life %s" % classtype] + ["mnist_train %s " % classtype])
    plt.xlim(xmin=0, xmax=300)
    plt.ylim(ymin=0, ymax=600)
    ax1.title.set_text("Class %s" % i)
    plt.xlabel('Number of 1 bits')
    plt.ylabel('Number of Images')
plt.subplots_adjust(wspace=0.3)
plt.show()

###################
# use a pretrained classifier to judge the images and plot prediction scores
##################################################################
##################################################################
##################################################################
##################################################################
##################################################################

"""
model and code from pytorch playground repo
"""
model_urls = {
    'mnist': 'http://ml.cs.tsinghua.edu.cn/~chenxi/pytorch-models/mnist-b07bb66b.pth'
}


class MLP(nn.Module):

    def __init__(self, input_dims, n_hiddens, n_class):
        super(MLP, self).__init__()
        assert isinstance(input_dims, int), 'Please provide int for input_dims'
        self.input_dims = input_dims
        current_dims = input_dims
        layers = OrderedDict()

        if isinstance(n_hiddens, int):
            n_hiddens = [n_hiddens]
        else:
            n_hiddens = list(n_hiddens)
        for i, n_hidden in enumerate(n_hiddens):
            layers['fc{}'.format(i + 1)] = nn.Linear(current_dims, n_hidden)
            layers['relu{}'.format(i + 1)] = nn.ReLU()
            layers['drop{}'.format(i + 1)] = nn.Dropout(0.2)
            current_dims = n_hidden
        layers['out'] = nn.Linear(current_dims, n_class)

        self.model = nn.Sequential(layers)

    def forward(self, input):
        assert input.size(1) == self.input_dims, "%s vs %s" % (input.size(), self.input_dims)
        return torch.softmax(self.model.forward(input), 1)


def mnist(input_dims=784, n_hiddens=[256, 256], n_class=10, pretrained=None):
    model = MLP(input_dims, n_hiddens, n_class)
    if pretrained is not None:
        m = model_zoo.load_url(model_urls['mnist'])
        state_dict = m.state_dict() if isinstance(m, nn.Module) else m
        assert isinstance(state_dict, (dict, OrderedDict)), type(state_dict)
        model.load_state_dict(state_dict)
    return model


model = mnist(pretrained=True)
model = model.eval()
still_classes = split_by_class(still)
real_classes = split_by_class(real)
still_classes_predictions = dict()
real_classes_predictions = dict()
for digit in still_classes:
    data = still_classes[digit]

    # make a batch of data for the whole class, predict and add to dict
    batch = np.concatenate([np.reshape(img, [1, -1]) for name, img in data], axis=0)
    batch = batch.astype(np.float32)

    prediction = model(torch.tensor(batch))
    prediction = prediction[:, int(digit)]
    prediction = prediction.detach().numpy()
    prediction = list(prediction)

    still_classes_predictions[str(digit)] = prediction

for digit in real_classes:
    data = real_classes[digit]

    # make a batch of data for the whole class, predict and add to dict
    batch = np.concatenate([np.reshape(img, [1, -1]) for name, img in data], axis=0)
    batch = batch.astype(np.float32)

    prediction = model(torch.tensor(batch))
    prediction = prediction[:, int(digit)]
    prediction = prediction.detach().numpy()
    prediction = list(prediction)

    real_classes_predictions[str(digit)] = prediction

fig = plt.figure(figsize=(20, 20))
fig.suptitle("Histograms of predictions on the corresponding correct class from a pretrained NN,\n real binarized "
             "data vs still life data")

for i in range(10):
    ax1 = fig.add_subplot(2, 5, i + 1)

    classtype = str(i)
    to_plot = [still_classes_predictions[classtype]] + [real_classes_predictions[classtype]]
    n, bins, patches = plt.hist(to_plot[::-1], bins=200, alpha=0.8, histtype='stepfilled')
    plt.legend(["mnist_still_life %s" % classtype] + ["mnist_train %s " % classtype])
    # plt.xlim(xmin=0, xmax=300)
    plt.ylim(ymin=0, ymax=300)
    ax1.title.set_text("Class %s" % i)
    plt.xlabel('Prediction/score of correct class')
    plt.ylabel('Number of Images')
plt.subplots_adjust(wspace=0.5)
plt.show()

###################
# make an album of supposedly good images
##################################################################
##################################################################
##################################################################
##################################################################
##################################################################

# name, data, total ones, prediction all in one place, for each class
stillNDP = {str(i): [] for i in range(10)}
realNDP = {str(i): [] for i in range(10)}

for digit in range(10):
    digit = str(digit)
    for nd, prediction in zip(still_classes[digit], still_classes_predictions[digit]):
        name, data = nd
        stillNDP[digit].append((name, data, np.sum(data), prediction))

    for nd, prediction in zip(real_classes[digit], real_classes_predictions[digit]):
        name, data = nd
        realNDP[digit].append((name, data, np.sum(data), prediction))

os.makedirs("mnist_still_dataset_10000", exist_ok=True)
os.system("rm mnist_still_dataset_10000/*")
for digit in range(10):
    digit = str(digit)
    stillNDP[digit] = sorted(stillNDP[digit], key=lambda x: x[3], reverse=True)
    for sample in stillNDP[digit][:1000]:
        scipy.misc.imsave("mnist_still_dataset_10000/%s" % sample[0], sample[1])

    print("digit %s : %s" % (digit, stillNDP[digit][999][3]))

# sample and check samples passing a threshold
threshold = 0.50
sampleN = 30
for digit in range(10):
    digit = str(digit)
    stillNDP[digit] = list(filter(lambda x: x[3] >= threshold, stillNDP[digit]))
    realNDP[digit] = list(filter(lambda x: x[3] >= threshold, realNDP[digit]))

fig = plt.figure(figsize=(20, 20))

title = "Sample pictures for still data with prediction score >= %s" % threshold
title += "\nTotal available digits passing threshold: %s" % sum(len(v) for k, v in stillNDP.items())
title += "\nTotal available digits passing threshold by class:"
for digit in range(10):
    digit = str(digit)
    title += " %s : %s |" % (digit, len(stillNDP[digit]))
fig.suptitle(title)
i = 1

for digit in range(10):
    digit = str(digit)
    imgs = stillNDP[digit]
    random.shuffle(imgs)

    sample = [sample[1] for sample in imgs[:sampleN]]

    for img in sample:
        ax1 = fig.add_subplot(10, sampleN, i)
        plt.imshow(img)
        plt.axis("off")
        i += 1
plt.subplots_adjust(wspace=0., hspace=0.)
plt.show()
