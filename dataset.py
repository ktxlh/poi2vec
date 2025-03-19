import torch
import numpy as np
import random

class Data():
    def __init__(self):

        self.id2route = None
        self.id2lr = None
        self.id2prob = None

        self.context_train = None
        self.target_train = None
        self.context_valid = None
        self.target_valid = None
        self.context_test = None
        self.target_test = None
        self.maxlen_context = 32

    def load(self):
        print("Loading data...")
        self.id2route = torch.from_numpy(np.load("./npy/id2route.npy"))
        self.id2lr = torch.from_numpy(np.load("./npy/id2lr.npy"))
        self.id2prob = torch.from_numpy(np.load("./npy/id2prob.npy"))
        
        self.context_train = torch.from_numpy(np.load("./npy/train_context.npy"))
        self.target_train = torch.from_numpy(np.load("./npy/train_target.npy"))
        self.context_valid = torch.from_numpy(np.load("./npy/valid_context.npy"))
        self.target_valid = torch.from_numpy(np.load("./npy/valid_target.npy"))
        print("Train/Valid/POI: {:d}/{:d}/{:d}".format(len(self.target_train), len(self.target_valid), len(self.id2route)))
        print("==================================================================================")

        return len(self.id2route)

    def train_batch_iter(self, batch_size):
        data = list(zip(self.context_train, self.target_train))
        random.shuffle(data)
        return self.batch_iter(data, batch_size)

    def valid_batch_iter(self, batch_size):
        data = list(zip(self.context_valid, self.target_valid))
        return self.batch_iter(data, batch_size)

    def batch_iter(self, data, batch_size):
        data_size = float(len(data))
        num_batches = int(np.ceil(data_size / batch_size))
        for batch_num in range(num_batches):
            start_index = int(batch_num * batch_size)
            end_index = min(int((batch_num + 1) * batch_size), int(data_size))
            yield data[start_index:end_index]
