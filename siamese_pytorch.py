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
        self.transform = transform
        self.target_transform = target_transform

        #Init as MNIST
        # It will be train or test set depending on passed params

        mnist_set = MNIST(root, train, transform, target_transform, download)


        if self.train:

            # Load explicitly from processed MNIST files created
            train_data, train_labels = torch.load(os.path.join(mnist_set.root, mnist_set.processed_folder, mnist_set.training_file))

            # To then pass to new functions
            train_labels_class, train_data_class = self.group_by_digit(train_data, train_labels)

            self.train_data, self.train_labels = self.generate_balanced_pairs(train_labels_class,train_data_class)

            #print(self.train_data.shape)

        else:

            test_data, test_labels = torch.load(os.path.join(mnist_set.root, mnist_set.processed_folder, mnist_set.test_file))

            test_labels_class, test_data_class = self.group_by_digit(test_data, test_labels)

            self.test_data, self.test_labels = self.generate_balanced_pairs(test_labels_class, test_data_class)

            #print(self.test_data.shape)


    """
    Isolating only methods that are actually different from base MNIST class in Pytorch
    """

    def __getitem__(self, index):

        if self.train:

            imgs, target = self.train_data[index], self.train_labels[index]

        else:
            imgs, target = self.test_data[index], self.test_labels[index]

        img_ar = []
        for i in range(len(imgs)):

            img = Image.fromarray(imgs[i].numpy(), mode='L')

            if self.transform is not None:

                img = self.transform(img)

            img_ar.append(img)

        if self.target_transform is not None:

            target = self.target_transform(target)


        return img_ar, target


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

                other_cls = [y for y in range(10) if y != i]

                rnd_cls = random.choice(other_cls)

                rnd_dist = random.randint(0, 100)

                #Append one positive example followed by one negative example
                data.append(torch.stack([data_class[i][j], data_class[i][j + rnd_dist], data_class[rnd_cls][j]]))

                #Append the pos neg labels for the two pairs
                labels.append([1, 0])


        return torch.stack(data), torch.tensor(labels)


#TODO: change to reproduce NormXCorr model by Submariam et al. (2016)
class Net(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 64, 7)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(64, 128, 5)
        self.conv3 = nn.Conv2d(128, 256, 5)
        self.linear1 = nn.Linear(2304, 512)

        self.linear2 = nn.Linear(512, 2)

    def forward(self, data):
        res = []
        for i in range(2):  # Siamese nets; sharing weights
            x = data[i]
            x = self.conv1(x)
            x = F.relu(x)
            x = self.pool1(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = self.conv3(x)
            x = F.relu(x)

            x = x.view(x.shape[0], -1)
            x = self.linear1(x)
            res.append(F.relu(x))

        res = torch.abs(res[1] - res[0])
        res = self.linear2(res)
        return res



"""
Series of methods for the learning settings
train/test/one-shot learning
"""

def train(model, device, train_loader, epoch, optimizer):

    #Sets the module in training mode
    model.train()



    for batch_idx, (data, target) in enumerate(train_loader):

        #Load data points on device
        for i in range(len(data)):

            data[i] = data[i].to(device)

        #Gradients are (re)set to zero at the beginning of each epoch
        optimizer.zero_grad()

        #Take first two images in each triple, i.e., positive pair
        output_positive = model(data[:2])
        #Take first and third image in each triple, i.e., negative pair
        output_negative = model(data[0:3:2])

        target = target.type(torch.LongTensor).to(device)

        #First label position, = positive g.t., = 1
        target_positive = torch.squeeze(target[:, 0])

        #Second label position, = negative g.t., = 0
        target_negative = torch.squeeze(target[:, 1])

        # L+, i.e., seeking 1.0 for positive examples
        loss_positive = F.cross_entropy(output_positive, target_positive)
        # L-, i.e., seeking 0.0 for negative examples
        loss_negative = F.cross_entropy(output_negative, target_negative)

        loss = loss_positive + loss_negative

        loss.backward()

        optimizer.step()

        #Log status every 10 epochs
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * batch_size, len(train_loader.dataset),
                       100. * batch_idx * batch_size / len(train_loader.dataset),
                loss.item()))


def test(model, device, test_loader):
    model.eval()

    with torch.no_grad():
        accurate_labels = 0
        all_labels = 0
        loss = 0
        for batch_idx, (data, target) in enumerate(test_loader):
            for i in range(len(data)):
                data[i] = data[i].to(device)

            output_positive = model(data[:2])
            output_negative = model(data[0:3:2])

            target = target.type(torch.LongTensor).to(device)
            target_positive = torch.squeeze(target[:, 0])
            target_negative = torch.squeeze(target[:, 1])

            loss_positive = F.cross_entropy(output_positive, target_positive)
            loss_negative = F.cross_entropy(output_negative, target_negative)

            loss = loss + loss_positive + loss_negative

            accurate_labels_positive = torch.sum(torch.argmax(output_positive, dim=1) == target_positive).cpu()
            accurate_labels_negative = torch.sum(torch.argmax(output_negative, dim=1) == target_negative).cpu()

            accurate_labels = accurate_labels + accurate_labels_positive + accurate_labels_negative
            all_labels = all_labels + len(target_positive) + len(target_negative)

        accuracy = 100. * accurate_labels / all_labels


        print('Test accuracy: {}/{} ({:.3f}%)\tLoss: {:.6f}'.format(accurate_labels, all_labels, accuracy, loss))



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

    #Model is defined and passed to either GPU or CPU device
    model = Net().to(device)

    import os

    if not os.path.isdir('./pt_results'):

        os.mkdir('pt_results')


    if do_learn:  # training mode

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

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        for epoch in range(num_epochs):

            train(model, device, train_loader, epoch, optimizer)
            test(model, device, test_loader)

            if epoch & save_frequency == 0:

                #Here it saves model at each epoch
                torch.save(model.state_dict(), 'pt_results/siamese_{:03}.pt'.format(epoch))


if __name__ == '__main__':

    main()