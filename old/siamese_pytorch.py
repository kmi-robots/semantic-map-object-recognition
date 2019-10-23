import os
from PIL import Image
import random
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from collections import OrderedDict
import time



do_learn = True
save_frequency = 2
batch_size = 16
lr = 0.05
num_epochs = 10
lr_decay = 0.0001
weight_decay = 0.0005
momentum = 0.9


#Variables and models to be loaded on GPU or CPU?
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BalancedMNIST(MNIST):

    """
    Extending MNIST Dataset class
    For each sample (anchor) randomly chooses a positive and negative samples

    """
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

            self.train_data, self.train_labels = self.generate_balanced_triplets(train_labels_class,train_data_class)

            #print(self.train_data.shape)

        else:

            test_data, test_labels = torch.load(os.path.join(mnist_set.root, mnist_set.processed_folder, mnist_set.test_file))

            test_labels_class, test_data_class = self.group_by_digit(test_data, test_labels)

            self.test_data, self.test_labels = self.generate_balanced_triplets(test_labels_class, test_data_class)

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

    def __len__(self):

        if self.train:

            return len(self.train_data)

        else:

            return len(self.test_data)

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
            data_class.append(torch.index_select(data, 0, indices))

        return labels_class, data_class


    def generate_balanced_triplets(self, labels_class, data_class):

        print(labels_class)
        print(data_class)

        data = []
        labels = []

        #Uncomment the following to check number of samples per class
        min_ = min([x.shape[0] for x in labels_class])

        #Check here for different sample number
        for i in range(10):

            print(i)

            for j in range(min_): #500  # create 500*10 triplets

                print(j)

                # choose random class different from current one

                other_cls = [y for y in range(10) if y != i]

                rnd_cls = random.choice(other_cls)

                rnd_dist= 0

                while(rnd_dist==j):

                    rnd_dist = random.randint(0,min_)

                #Append one positive example followed by one negative example
                data.append(torch.stack([data_class[i][j], data_class[i][rnd_dist], data_class[rnd_cls][j]]))

                #Append the pos neg labels for the two pairs
                labels.append([1, 0])

        #print(torch.stack(data).shape)


        return torch.stack(data), torch.tensor(labels)




#Reproducing NormXCorr model by Submariam et al. (NIPS 2016)
from torch.autograd import Function

import ncc  #Our built extension


class normxcorr(Function):

    @staticmethod
    def forward(ctx, X_, Y_, patch_size=5, stride=1, epsilon=0.01):

        ctx.saved_for_backward = [X_, Y_]
        ctx.patch_size = patch_size
        ctx.stride = stride
        ctx.epsilon = epsilon

        return ncc.forward(X_, Y_, ctx.patch_size, ctx.stride, ctx.epsilon)

    @staticmethod
    def backward(ctx, grad_output):

        input1, input2 = ctx.saved_for_backward

        #grad_input1 = torch.zeros(input1.size()).to(device)
        #grad_input2 = torch.zeros(input2.size()).to(device)

        nan_idxs = torch.nonzero(torch.isnan(grad_output))

        # Print warning in case any NaN is found
        if nan_idxs.nelement() != 0:
            print("NaN value found in grad_output ")

        # grad_input = [derivate forward(input) wrt parameters] * grad_output
        grad_input1, grad_input2 = ncc.backward(input1,
                                                 input2,
                                                grad_output,
                                                ctx.patch_size,
                                                ctx.stride,
                                                ctx.epsilon)



        nan_idxs = torch.nonzero(torch.isnan(grad_input1))

        # Print warning in case any is found
        if nan_idxs.nelement() != 0:
            print("NaN value found in grad_input1 ")

        nan_idxs = torch.nonzero(torch.isnan(grad_input2))

        # Print warning in case any is found
        if nan_idxs.nelement() != 0:
            print("NaN value found in grad_input2")

        return grad_input1, grad_input2






"""
Series of methods for the learning settings
train/test/one-shot learning
"""

"""
def weights_init(module):

    for name in module.named_children():

        #print(key)

        if 'sequential' in name:

           print(dict(module.get_index_by(name).named_children()).keys())

    \"""
    classname = self.__class__.__name__

    if module.sequential.keys().find('conv') != -1:
        
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    \"""
"""



def train(model, train_loader, epoch, optimizer):

    #Sets the module in training mode
    model.train()

    accurate_labels = 0
    all_labels = 0

    for batch_idx, (data, target) in enumerate(train_loader):

        print("Currently processing batch no %i" % batch_idx)

        # Data is a list of three image batches defined as before
        #Load data points on device
        for i in range(len(data)):

            data[i] = data[i].to(device, non_blocking=True)

        #Gradients are (re)set to zero at the beginning of each epoch
        optimizer.zero_grad()

        #Take first two images in each triple, i.e., positive pair
        output_positive, out_pos_soft = model(data[:2])

        check_for_NaNs(model)
        #print("Checked for NaNs")

        #Take first and third image in each triple, i.e., negative pair
        output_negative, out_neg_soft = model(data[0:3:2])

        check_for_NaNs(model)
        #print("Checked for NaNs")


        print("Output positive")
        print(output_positive)
        print(out_pos_soft)
        print("Output negative")
        print(torch.nn.Softmax(output_positive))
        print(out_neg_soft)

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

        #start = time.time()
        #print("Starting back prop")
        loss.backward()
        #print("Backward complete in %f seconds" % float(time.time()-start))

        optimizer.step()

        check_for_NaNs(model)

        accurate_labels_positive = torch.sum(torch.argmax(output_positive, dim=1) == target_positive).cpu()
        accurate_labels_negative = torch.sum(torch.argmax(output_negative, dim=1) == target_negative).cpu()

        accurate_labels = accurate_labels + accurate_labels_positive + accurate_labels_negative
        all_labels = all_labels + len(target_positive) + len(target_negative)



        #Log status every 10 batches
        #if batch_idx % 10 == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * batch_size, len(train_loader.dataset),
            100. * batch_idx * batch_size / len(train_loader.dataset),
            loss.item()))

    accuracy = 100. * accurate_labels / all_labels
    print("Training accuracy after epoch: %f" % accuracy)



def test(model, test_loader):

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

    if device == 'cuda':

        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        from torch.backends import cudnn
        cudnn.benchmark = True

    """
    Pytorch transforms passed to the data Loader
    
    1) The first transform converts a PIL Image or numpy ndarray in the [0, 255] range
    into a torch.FloarTensor in range [0.0, 1.0]
    2) It is here combined with a Normalize transform, normalizing each input channel
    w.r.t. mean 0.5 and std dev 1.0
    ADDED: 3) transforming to 3-channel images. Placed first as PIL Images needed 
    4) Resizing to 160x60
    """

    trans = transforms.Compose([transforms.Resize((160,60)),
                                transforms.Grayscale(3),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,),(1.0,))])

    #Model is defined and passed to either GPU or CPU device
    model = Net().to(device, non_blocking=True )

    #weights_init(model)

    #Create subfolder to store results, if not already there
    import os

    if not os.path.isdir('./data'):

        os.mkdir('data')

    if not os.path.isdir('./pt_results'):

        os.mkdir('pt_results')


    if do_learn:  # training mode

        """
        Train and test data will be downloaded and normalized
        """

        train_loader = torch.utils.data.DataLoader(
            BalancedMNIST('./data', train=True, download=True, transform=trans), batch_size=batch_size,
            shuffle=True)

        test_loader = torch.utils.data.DataLoader(
            BalancedMNIST('./data', train=False, download=True, transform=trans), batch_size=batch_size,
            shuffle=False)

        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)

        for epoch in range(num_epochs):

            train(model, train_loader, epoch, optimizer)
            test(model, test_loader)

            if epoch & save_frequency == 0:

                #Here it saves model

                torch.save(model.state_dict(), 'pt_results/siamese_{:03}.pt'.format(epoch))


if __name__ == '__main__':

    main()