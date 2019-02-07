import os
from PIL import Image
import random
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
import itertools


do_learn = True
save_frequency = 2
batch_size = 16
lr = 0.001
num_epochs = 10
weight_decay = 0.0001


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


    def generate_balanced_triplets(self, labels_class, data_class):

        data = []
        labels = []

        #Uncomment the following to check number of samples per class
        #print([x.shape[0] for x in labels_class])

        #Check here for different sample number
        for i in range(10):

            for j in range(500):  # create 500*10 triplets

                # choose random class different from current one

                other_cls = [y for y in range(10) if y != i]

                rnd_cls = random.choice(other_cls)

                rnd_dist = random.randint(0, 100)

                #Append one positive example followed by one negative example
                data.append(torch.stack([data_class[i][j], data_class[i][j + rnd_dist], data_class[rnd_cls][j]]))

                #Append the pos neg labels for the two pairs
                labels.append([1, 0])

        #print(torch.stack(data).shape)

        return torch.stack(data), torch.tensor(labels)


#Reproducing NormXCorr model by Submariam et al. (NIPS 2016)

class NormXCorr(nn.Module):

    def __init__(self):

        super().__init__()


    def forward(self, data, patch_size=5, stride=1):

        """
        - data[1] output of one pipeline
        - data[0] output of the other

        Following the paper terminology:
        data[1] = feature map X
        data[0] = feature map Y

        Except here grouped in batches
        """

        X = data[0] #.double() #Trying float64 in case more precision needed
        Y = data[1] #.double()


        X_ = torch.tensor(X)
        Y_ = torch.tensor(Y)

        in_depth = X.shape[1]
        in_height= X.shape[2]
        in_width= X.shape[3]

        out_depth= patch_size*in_width*in_depth

        d = int(patch_size/2)


        """
            for each depth i in range(25):

                1. take the ith 37x12 feature map from X
                    1.a create copy of X with padding of two at each margin -> 41x16
                2. take the ith 37x12 feature map from Y
                    2.a create copy of Y with same size of 1.a, but extra vertical padding of two each -> 45x16            
        """

        output = []

        for i in range(in_depth):

            X_i = X_[:, i, :]
            Y_i = Y_[:, i, :]

            # print(Y_i.shape) #batch_sizexHxW, e.g., 16x37x12
            #print(X_i.shape)

            X_pad =F.pad(X_i, (d, d, d, d))
            Y_pad = F.pad(Y_i, (d, d, 2*d, 2*d))

            #Empty matrix to preserve original positions

            M_i = torch.zeros([batch_size, in_width*patch_size, in_height, in_width], dtype=torch.float32)

            #3.  For each jth pixel in X:
            for y,x in itertools.product(range(d, in_height+d), range(d,in_width+d)):

                y_orig = y-d
                x_orig = x-d

                #3.a take related 5x5 patch in 1.a (E)

                E_ = X_pad[:, y-d:y+d+1, x-d:x+d+1]

                #print(E_.shape)

                # Wider rectangular (inexact) search area, width is fixed at 12
                #rect_area = Y_pad[:, y:y+2*d+1, d:in_width+d]

                #Extra padding of 2 to extract 5x5 patches
                padded_width = Y_pad.shape[1]
                rect_area = Y_pad[:, y-d:y + 3*d + 1, d-2:padded_width+1]
                #print(rect_area.shape)

                # 3.b take all possible 5x5 patches in 2.a. (all possible Fs)

                F_values = rect_area.unfold(1, patch_size, stride).unfold(2, patch_size, stride)

                #print(F_values.shape)

                #3.c Init empty output vec of (area,1), where area = rect_area(3.b) -> (60,1)

                normxcorrs = self.compute_normxcorr(E_, F_values, patch_size, in_width)

                # print(normxcorr_values.shape) #batch_sizex1 , i.e., each similarity value is a scalar
                # stack normxcorr_values, there will be 60 in the end

                normxcorrs = torch.stack(normxcorrs, 1)

                #print(normxcorrs.shape) # shape batch_sizex 60 x 1

                M_i[:, :, y_orig, x_orig] = normxcorrs  # Assign to original position in feature map

            #for all y and x
            # This for ends with a 37x12x60 matrix
            output.append(M_i)

        #print(len(output))  #25
        #for all depths, yielding output -> 37x12x60*25 = 37x12x1500
        out = torch.stack(output, 1)

        return out.reshape((batch_size, out_depth,in_height,in_width))

    def compute_normxcorr(self, E_, F_values, patch_size, in_width, epsilon=0.01):

        """
            3.d for each E,F pair:
                - reshape both to be (25,) --> E',F'
                - compute mean of E' and F'
                - compute std devs of E' and F'
                - add 0.01 to each std dev to avoid division by 0
                - compute normcrosscorr between E'F'
        """
        normxcorrs = []

        E_ = E_.reshape((batch_size, E_.shape[1] * E_.shape[2]))

        # For each patch in rectangular search window: 5x12 patches of size 5x5 each

        for y__, x__ in itertools.product(range(patch_size), range(in_width)):

            F_ = F_values[:, y__, x__, : ]

            #print(F_.shape)  # batch_sizex5x5
            #print(E_.shape)

            F_ = F_.reshape((batch_size, F_.shape[1] * F_.shape[2]))

            #print(F_.shape)  # batch_sizex25, i.e., N-dimensional vector

            # Batch-wise computations
            E_mean = E_.view(batch_size, -1).mean(1, keepdim=True)
            F_mean = E_.view(batch_size, -1).mean(1, keepdim=True)
            # print(E_mean.shape) #batch_sizex1

            E_std = E_.view(batch_size, -1).std(1, keepdim=True)
            F_std = F_.view(batch_size, -1).std(1, keepdim=True)
            E_std = E_std + epsilon
            F_std = F_std + epsilon
            # print(E_std.shape) #batch_sizex1

            E_norm = (E_ - E_mean) / (E_std * E_.shape[1])
            F_norm = (F_ - F_mean) / F_std

            #print((E_norm * F_norm).shape)

            normxcorrs.append(torch.sum(E_norm * F_norm, 1))

        return normxcorrs



#The whole feed-forward architecture
class Net(nn.Module):

    def __init__(self):

        super().__init__()
        """
        All variables defined here are ultimately
        passed as params to the optimizer
        
        """

        #Redefining first part as a sequential model
        self.sequential = nn.Sequential(

            nn.Conv2d(3, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(20, 25, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.normxcorr = NormXCorr()

        self.normrelu = nn.ReLU()

        self.dimredux = self.dimredux = nn.Sequential(

            nn.Conv2d(1500, 25, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(25, 25, kernel_size=3),
            nn.MaxPool2d(2)
        )

        self.linear1 = nn.Linear(25*17*5, 500)

        self.linear2 = nn.Linear(500, 2)

        #self.softmax = nn.Softmax()


    def forward(self, input):


        res = []

        # Defines the two pipelines
        for i in range(2):  # Siamese nets; sharing weights

            x = input[i]
            #print(x.shape)
            res.append(self.sequential(x))
            #print(self.sequential(x).shape)


        res = self.normxcorr(res) # batch_size x 1500 x 37 x 12

        res = self.normrelu(res)

        res = self.dimredux(res) # batch_size x 25 x 17 x 5

        res =  res.view(res.size()[0], -1) # (batch_size. 2125) , i.e. flattened

        res = self.linear1(res) # batch_size x 2121 x 500

        res = self.linear2(res) # batch_size x 500 x 2

        #print(res.shape)
        #res = self.softmax(res) #Calculated in train loop later

        return res


"""
Series of methods for the learning settings
train/test/one-shot learning
"""

def train(model, device, train_loader, epoch, optimizer):

    #Sets the module in training mode
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):

        # Data is a list of three image batches defined as before
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

        #Log status every 10 batches
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
    ADDED: 3) transforming to 3-channel images. Placed first as PIL Images needed 
    4) Resizing to 160x60
    """

    trans = transforms.Compose([transforms.Resize((160,60)),
                                transforms.Grayscale(3),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,),(1.0,))])

    #Model is defined and passed to either GPU or CPU device
    model = Net().to(device)

    #Create subfolder to store results, if not already there
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