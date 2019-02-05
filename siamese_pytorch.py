import codecs
import errno
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import random
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from tqdm import tqdm

do_learn = True
save_frequency = 2
batch_size = 16
lr = 0.001
num_epochs = 10
weight_decay = 0.0001



class BalancedMNIST(MNIST):

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):


        self.train = train

        #Init as MNIST
        # It will be train or test set depending on passed params

        mnist_set = MNIST(root, train, transform, target_transform, download)


        if self.train:

            # Load explicitly from processed MNIST files created
            train_data, train_labels = torch.load(os.path.join(mnist_set.root, mnist_set.processed_folder, mnist_set.training_file))

            # To then pass to new functions
            train_labels_class, train_data_class = self.group_by_digit(train_data, train_labels)

            self.train_data, self.train_labels = self.generate_balanced_pairs(train_labels_class,train_data_class)

            print(self.train_data.shape)

        else:

            test_data, test_labels = torch.load(os.path.join(mnist_set.root, mnist_set.processed_folder, mnist_set.test_file))

            test_labels_class, test_data_class = self.group_by_digit(test_data, test_labels)

            self.test_data, self.test_labels = self.generate_balanced_pairs(test_labels_class, test_data_class)

            print(self.test_data.shape)


    """
    Isolating only methods that are actually different from base MNIST class in Pytorch
    """

    def group_by_digit(self, data, labels):

        """
        Returns lists of len 10 grouping tensors
        belonging to the same digit/class at
        each indexed position

        """
        labels_class = []
        data_class = []

        # For each digit in the data
        for i in range(10):
            # Check location of data labeled as current digit
            # and reduce to one-dimensional LongTensor
            indices = torch.squeeze((labels == i).nonzero())

            # Use produced indices to select rows in the original data tensors loaded
            # And add them to related list
            labels_class.append(torch.index_select(labels, 0, indices))
            data_class.append(torch.index_select(data, 0, indices))

        return labels_class, data_class


    def generate_balanced_pairs(self, labels_class, data_class):

        data = []
        labels = []

        #Uncomment the following to check number of samples per class
        #print([x.shape[0] for x in labels_class])

        #Check here for different sample number
        for i in range(10):

            for j in range(500):  # create 500*10 pairs

                # choose random class different from current one

                other_cls = [y for y in range(10) if y!=i]

                rnd_cls = random.choice(other_cls)

                rnd_dist = random.randint(0, 100)

                #Append one positive example followed by one negative example
                data.append(torch.stack([data_class[i][j], data_class[i][j + rnd_dist], data_class[rnd_cls][j]]))

                #Append the pos neg labels for the two pairs
                labels.append([1, 0])

        return torch.stack(data), torch.tensor(labels)



def main():

    #If cuda device not available it runs of GPU
    #major difference with Keras

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    """
    Pytorch transforms passed to the data Loader
    
    1) The first transform converts a PIL Image or numpy ndarray in the [0, 255] range
    into a torch.FloarTensor in range [0.0, 1.0]
    2) It is here combined with a Normalize transform, normalizing each input channel
    w.r.t. mean 0.5 and std dev 1.0
    """

    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])

    """
    Train and test data will be downloaded and normalized
    """

    #BalancedMNIST('./data', train=True, download=True, transform=trans)
    #BalancedMNIST('./data', train=False, download=True, transform=trans)


    train_loader = torch.utils.data.DataLoader(
        BalancedMNIST('./data', train=True, download=True, transform=trans), batch_size=batch_size,
        shuffle=True)


    test_loader = torch.utils.data.DataLoader(
        BalancedMNIST('./data', train=False, download=True, transform=trans), batch_size=batch_size,
        shuffle=False)





if __name__ == '__main__':

    main()