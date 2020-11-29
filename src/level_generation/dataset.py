import os
import numpy as np
import random

class Dataset:
    def __init__(self, samples_path, labels_path, train_perc=0.7):
        assert train_perc >= 0 and train_perc <= 1

        self.samples_path = samples_path
        self.labels_path = labels_path

        # assert that files are loaded ordered and correctly!
        assert os.listdir(samples_path) == os.listdir(labels_path)

        self.samples_file_list = [os.path.join(self.samples_path, filename) for filename in os.listdir(
            samples_path) if filename.endswith(".npy")]
        self.labels_file_list = [os.path.join(self.labels_path, filename) for filename in os.listdir(
            labels_path) if filename.endswith(".npy")]

        # loading data
        self.samples_data = [self._load_file(
            filename) for filename in self.samples_file_list]
        self.labels_data = [self._load_file(
            filename) for filename in self.labels_file_list]

        self.index_training = 0
        self.index_test = 0

        # shuffle data before dividing between test and train
        self.samples_data, self.labels_data = self._shuffle(
            self.samples_data, self.labels_data)

        self.samples_data_training = self.samples_data[:int(
            len(self.samples_data) * train_perc)]
        self.samples_data_test = self.samples_data[int(
            len(self.samples_data) * train_perc):]

        self.labels_data_training = self.labels_data[:int(
            len(self.labels_data) * train_perc)]
        self.labels_data_test = self.labels_data[int(
            len(self.labels_data) * train_perc):]

    def init_training(self):
        self.index_training = 0
        self._shuffle_train()

    def init_test(self):
        self.index_test = 0
        self._shuffle_test()

    def _shuffle(self, a, b):
        c = list(zip(a, b))
        random.shuffle(c)
        return zip(*c)

    def _shuffle_train(self):
        self.samples_data_training, self.labels_data_training = self._shuffle(
            self.samples_data_training, self.labels_data_training)

    def _shuffle_test(self):
        self.samples_data_test, self.labels_data_test = self._shuffle(
            self.samples_data_test, self.labels_data_test)

    def _load_file(self, filepath):
        return np.load(filepath).astype('float32')

    def has_next_training(self):
        return self.index_training < len(self.samples_data_training) and self.index_training < len(self.labels_data_training)

    def has_next_test(self):
        return self.index_test < len(self.samples_data_test) and self.index_test < len(self.labels_data_test)

    def _next(self, batch_size, index, data, label):
        if index >= len(data) or index >= len(label):
            return None
        res = (data[index: min(index + batch_size, len(data))],
               label[index: min(index + batch_size, len(data))])
        return np.asarray(res)

    def next_training(self, batch_size):
        res = self._next(batch_size, self.index_training, self.samples_data_training, self.labels_data_training)
        if res is not None:
            self.index_training += batch_size
        return res

    def next_test(self, batch_size):
        res = self._next(batch_size, self.index_test, self.samples_data_test, self.labels_data_test)
        if res is not None:
            self.index_test += len(res)
        return res

    def _one_hot(self, data, nb_classes):
        '''
        convert numpy level to one hot encoding
        '''
        flattened = data.reshape(-1)
        res = np.eye(nb_classes)[flattened]
        res = res.reshape(list(data.shape) + [nb_classes])
        return res
