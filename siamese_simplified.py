import numpy as np
import os
from PIL import Image
import random
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import transforms
from collections import OrderedDict
from torchvision.datasets import MNIST
import cv2

import visdom #For data viz

do_learn = True
save_frequency = 2
batch_size = 64
lr = 0.001
num_epochs = 2
weight_decay = 0.0001

vis = visdom.Visdom() #Define main session once


class BalancedTriplets(torch.utils.data.Dataset):

    """Generic class to load custom dataset and generate balanced triplets
       in the form <anchor, positive example, negative example>
       for both an image collection and a set of labels
    """
    #Local paths
    raw_folder = 'shapenet'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'



    def __init__(self, root, train=True, transform=None, target_transform=None):

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if not os.path.isdir(os.path.join(self.root, self.processed_folder)):
            os.mkdir(os.path.join(self.root, self.processed_folder))


        self.prep()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.')

        if self.train:

            # Load explicitly from processed MNIST files created
            train_data, train_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.training_file))

            # To then pass to new functions
            train_labels_class, train_data_class = group_by_class(train_data, train_labels)

            self.train_data, self.train_labels = generate_balanced_triplets(train_labels_class, train_data_class)


        else:

            test_data, test_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.test_file))

            test_labels_class, test_data_class = group_by_class(test_data, test_labels)

            self.test_data, self.test_labels = generate_balanced_triplets(test_labels_class, test_data_class)

            # print(self.test_data.shape)

    def __getitem__(self, index):

        if self.train:
            imgs, target = self.train_data[index], self.train_labels[index]
        else:
            imgs, target = self.test_data[index], self.test_labels[index]

        img_ar = []

        for i in range(len(imgs)):


            img = imgs[i] #Image.fromarray(imgs[i].numpy(), mode='L')
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

    def _check_exists(self):

        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
               os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))

    def prep(self):

        print('Reading images and labels...')

        #Load from local

        training_set = (self.read_files(os.path.join(self.root, self.raw_folder, 'train')))

        test_set = (self.read_files(os.path.join(self.root, self.raw_folder, 'test')))

        #Save as .pt files

        with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as f:

            torch.save(training_set, f)

        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:

            torch.save(test_set, f)

        print('Done!')

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))


        return fmt_str



    def read_files(self, path, total=100):


        class_ = 0

        data = torch.empty((total, 3, 160, 60))
        labels = torch.empty((total))

        #Subfolders are named after classes here
        iter = 0
        for root, dirs, files in os.walk(path):

            if files:

               for file in files:

                   data[iter,:] = torch.from_numpy(img_preproc(os.path.join(root, file)))

                   labels[iter] = torch.LongTensor([class_])

                   iter +=1

               class_ += 1


        return data, labels


def img_preproc(path_to_image):

    img = cv2.imread(path_to_image)
    x = cv2.resize(img, (60,160))
    x = np.asarray(x)
    #display_img(x)
    x = x.astype('float')
    #display_img(x)
    x = np.expand_dims(x, axis= 0)

    x = np.reshape(x, (x.shape[0], x.shape[3], x.shape[1], x.shape[2]))

    #display_img(x)
    return x/255.


def group_by_class(data, labels, classes=10):

    """
    Returns lists of len 10 grouping tensors
    belonging to the same digit/class at
    each indexed position

    """
    labels_class = []
    data_class = []



    # For each digit in the data
    for i in range(classes):
        # Check location of data labeled as current digit
        # and reduce to one-dimensional LongTensor

        indices = torch.squeeze((labels == i).nonzero())

        # Use produced indices to select rows in the original data tensors loaded
        # And add them to related list
        labels_class.append(torch.index_select(labels, 0, indices))
        data_class.append(torch.index_select(data, 0, indices))
        data_class.append(torch.index_select(data, 0, indices))

    return labels_class, data_class


def generate_balanced_triplets(labels_class, data_class):

    data = []
    labels = []

    # Uncomment the following to check number of samples per class
    min_ = min([x.shape[0] for x in labels_class])


    # Check here for different sample number
    for i in range(len(labels_class)):

        for j in range(min_):  # 500  # create 500*10 triplets

            r = [y for y in range(min_) if y != j]  # excluding j

            for idx in r:

                # choose random class different from current one
                other_cls = [y for y in range(len(labels_class)) if y != i]

                rnd_cls = random.choice(other_cls)
                rnd_idx = random.randint(0,min_-1)

                data.append(torch.stack([data_class[i][j], data_class[i][idx], data_class[rnd_cls][rnd_idx]]))
                #data.append(torch.stack([data_class[i][j], data_class[i][rnd_dist], data_class[rnd_cls][j]]))

                # Append the pos neg labels for the two pairs
                labels.append([1, 0])

    # print(torch.stack(data).shape)

    return torch.stack(data), torch.tensor(labels)


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
            train_data, train_labels= torch.load(os.path.join(mnist_set.root, mnist_set.processed_folder, mnist_set.training_file))

            # To then pass to new functions
            train_labels_class, train_data_class = group_by_class(train_data, train_labels)

            self.train_data, self.train_labels =generate_balanced_triplets(train_labels_class,train_data_class)

            #print(self.train_data.shape)

        else:

            test_data, test_labels = torch.load(os.path.join(mnist_set.root, mnist_set.processed_folder, mnist_set.test_file))

            test_labels_class, test_data_class = group_by_class(test_data, test_labels)

            self.test_data, self.test_labels = generate_balanced_triplets(test_labels_class, test_data_class)

            #print(self.test_data.shape)


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



class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.sequential = nn.Sequential(OrderedDict({
            'conv_1': nn.Conv2d(3, 20, kernel_size=5),
            'relu_1': nn.ReLU(),
            'maxpool_1': nn.MaxPool2d(2),
            'conv_2': nn.Conv2d(20, 25, kernel_size=5),
            'relu_2': nn.ReLU(),
            'maxpool_2': nn.MaxPool2d(2),
            'conv_3': nn.Conv2d(25, 30, kernel_size=5),
            'relu_3': nn.ReLU(),
            'maxpool_3': nn.MaxPool2d(2)
        }))

        self.linear1 = nn.Linear(16 * 4 * 30, 500)

        self.linear2 = nn.Linear(500, 2)

    def forward_once(self, x):

        #print(x.shape)
        x = self.sequential(x)
        #print(x.shape)
        #Flatten
        x= x.view(x.size(0), -1)
        #print(x.shape)

        return self.linear1(x)


    def forward(self, data):

        res = torch.abs(self.forward_once(data[1]) - self.forward_once(data[0]))
        res = self.linear2(res)
        return res


def train(model, device, train_loader, epoch, optimizer):

    model.train()

    accurate_labels = 0
    all_labels = 0
    epoch_loss = 0


    for batch_idx, (data, target) in enumerate(train_loader):

        #print(data[0].shape)
        for i in range(len(data)):
            data[i] = data[i].to(device)

        optimizer.zero_grad()
        output_positive = model(data[:2])
        output_negative = model(data[0:3:2])

        target = target.type(torch.LongTensor).to(device)
        target_positive = torch.squeeze(target[:, 0])
        target_negative = torch.squeeze(target[:, 1])

        #print("Output positive")
        #print(output_positive)
        #print("Output negative")
        #print(output_negative)


        loss_positive = F.cross_entropy(output_positive, target_positive)
        loss_negative = F.cross_entropy(output_negative, target_negative)
        #print(loss_positive)
        #print(loss_negative)
        loss = loss_positive + loss_negative

        loss.backward()

        optimizer.step()

        norm_loss_p = output_positive.shape[0] * loss_positive.item()
        norm_loss_n = output_negative.shape[0] * loss_negative.item()

        epoch_loss += norm_loss_p + norm_loss_n

        accurate_labels_positive = torch.sum(torch.argmax(output_positive, dim=1) == target_positive).cpu()
        accurate_labels_negative = torch.sum(torch.argmax(output_negative, dim=1) == target_negative).cpu()

        accurate_labels = accurate_labels + accurate_labels_positive + accurate_labels_negative
        all_labels = all_labels + len(target_positive) + len(target_negative)


    accuracy = 100 * accurate_labels / all_labels


    print("Epoch {}/{}, Loss: {:.3f}, Accuracy: {:.3f}%".format(epoch+1,num_epochs, epoch_loss,accuracy))

    return torch.Tensor([epoch_loss, accuracy])




def test(model, device, test_loader):
    model.eval()

    with torch.no_grad():
        accurate_labels = 0
        all_labels = 0
        epoch_loss = 0
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

            #loss = loss_positive + loss_negative

            norm_loss_p = output_positive.shape[0] * loss_positive.item() #as cross_entropy loss is the mean over batch size by default
            norm_loss_n = output_negative.shape[0] * loss_negative.item() #as cross_entropy  loss is the mean over batch size by default

            epoch_loss += norm_loss_p + norm_loss_n

            accurate_labels_positive = torch.sum(torch.argmax(output_positive, dim=1) == target_positive).cpu()
            accurate_labels_negative = torch.sum(torch.argmax(output_negative, dim=1) == target_negative).cpu()

            accurate_labels = accurate_labels + accurate_labels_positive + accurate_labels_negative
            all_labels = all_labels + len(target_positive) + len(target_negative)

        accuracy = 100. * accurate_labels / all_labels
        print('Test accuracy: {}/{} ({:.3f}%)\t Loss: {:.6f}'.format(accurate_labels, all_labels, accuracy, epoch_loss))



        return torch.Tensor([epoch_loss, accuracy])


def oneshot(model, device, data):
    model.eval()

    with torch.no_grad():
        for i in range(len(data)):
            data[i] = data[i].to(device)

        output = model(data)
        return torch.squeeze(torch.argmax(output, dim=1)).cpu().item()


def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    """

    trans = transforms.Compose([transforms.Resize((160, 60)),
                                transforms.Grayscale(3),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (1.0,))])

    """

    trans = transforms.Compose([
                                transforms.Normalize((0.5,), (1.0,))
                                ])

    model = Net().to(device)


    if do_learn:  # training mode

        train_loader = torch.utils.data.DataLoader(
           BalancedTriplets('/home/agni', train=True, transform=trans), batch_size=batch_size,
           shuffle=True)

        #train_loader = torch.utils.data.DataLoader(
        #   BalancedMNIST('../data', train=True, transform=trans), batch_size=batch_size,
        #    shuffle=True)


        test_loader = torch.utils.data.DataLoader(
            BalancedTriplets('/home/agni', train=False, transform=trans), batch_size=batch_size,
            shuffle=False)

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        epoch_train_metrics= torch.empty((num_epochs, 2))
        epoch_test_metrics = torch.empty_like(epoch_train_metrics)


        for epoch in range(num_epochs):

            epoch_train_metrics[epoch, :] = train(model, device, train_loader, epoch, optimizer)

            epoch_test_metrics[epoch, :] = test(model, device, test_loader)


            if epoch & save_frequency == 0:
                torch.save(model, 'pt_results/siamese_{:03}.pt'.format(epoch))


        print(epoch_train_metrics.shape)
        print(epoch_train_metrics[:,1].shape)
        print(epoch_train_metrics[:, 2].shape)
        print(torch.stack((epoch_train_metrics[:,1], epoch_test_metrics[:,1]), dim=1).shape)

        epoch_losses = torch.stack((epoch_train_metrics[:,:2], epoch_test_metrics[:,:2]), dim=1)
        epoch_accs = torch.stack((epoch_train_metrics[:, 0:3:2], epoch_test_metrics[:, 0:3:2]), dim=1)

        #Metric over epoch plots

        vis.line(Y=epoch_losses, X=epoch_losses[:,0], opts={
                'showlegend': True,
                'title': 'Loss per epoch plot'
                 }
                 )

    else:  # prediction
        prediction_loader = torch.utils.data.DataLoader(
            BalancedTriplets('/home/agni', train=False, transform=trans), batch_size=batch_size,
            shuffle=True)
        model.load_state_dict(torch.load(load_model_path))
        data = []
        data.extend(next(iter(prediction_loader))[0][:3:2])
        same = oneshot(model, device, data)
        if same > 0:
            print('These two images are of the same number')
        else:
            print('These two images are not of the same number')


if __name__ == '__main__':
    main()
