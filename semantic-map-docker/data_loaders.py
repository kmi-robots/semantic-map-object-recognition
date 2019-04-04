from torchvision.datasets import MNIST
import cv2
import numpy as np
import os
from PIL import Image
import torch
import random

""""

Classes and methods to create balanced triplets
- In a custom case on own data set
- In the case of a standard set, i.e., extending the MNIST class


"""
HANS = True

class BalancedTriplets(torch.utils.data.Dataset):

    """Generic class to load custom dataset and generate balanced triplets
       in the form <anchor, positive example, negative example>
       for both an image collection and a set of labels
    """
    #Local paths
    raw_folder = 'shapenet'
    processed_folder = 'processed'
    training_file = 'shapenet_training.pt'
    test_file = 'shapenet_test.pt'
    to_val = 'test'

    def __init__(self, root, train=True, transform=None, target_transform=None, Hans=HANS):

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if Hans:
            self.raw_folder = 'Hans-all'
            self.out_name = 'Hans-split'

            if not os.path.isdir(os.path.join(self.root, self.out_name)):
                self.split_data(os.path.join(self.root, self.raw_folder), os.path.join(self.root, self.out_name))

            self.raw_folder = self.out_name
            self.training_file = 'hans_training.pt'
            self.test_file = 'hans_test.pt'
            self.to_val = 'val'

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

            print(self.train_data.shape)
        else:

            test_data, test_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.test_file))

            test_labels_class, test_data_class = group_by_class(test_data, test_labels)

            self.test_data, self.test_labels = generate_balanced_triplets(test_labels_class, test_data_class)

            # print(self.test_data.shape)

    def split_data(self, data_path, out_path, ratio=(.8, .1, .1)):

        #TODO Add dependency in readme
        import split_folders
        # 80% train, 10% val, 10% test
        #random seeds for shuffling defaults to 1337
        split_folders.ratio(input=data_path, output=out_path, seed=1337, ratio=ratio)


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

        test_set = (self.read_files(os.path.join(self.root, self.raw_folder, self.to_val), train=False))

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



    def read_files(self, path,  train=True, ResNet=True, Hans=HANS, n=20):

        class_ = 0

        if train:

            total = 100

        else:
            total = 82

        if Hans:

            if train:
                total = 527
            else:
                total = 64

        if ResNet:

            data = torch.empty((total, 3, 224, 224))
        else:

            data = torch.empty((total, 3, 160, 60))

        labels = torch.empty((total))
        names = {}

        #Subfolders are named after classes here
        iteration = 0

        for root, dirs, files in os.walk(path):

            if files:

                classname = str(root.split('/')[-1])

                #NOTE: the assumption is that images are grouped in subfolders by class
                #example_no = 1
                
                if train and Hans:
                    sample = random.sample(files, n)

                else:
                    sample = files

                for file in sample:

                    img_tensor = torch.from_numpy(img_preproc(os.path.join(root, file)))
                    filename = str(file.split('/')[-1])

                    data[iteration, :] = img_tensor
                    labels[iteration] = torch.LongTensor([class_])

                    # ID = <classname_filename>
                    names[classname+'_'+filename] = img_tensor
                    #example_no += 1
                    iteration += 1

                class_ += 1

        #Save serialized object separately for IDs
        if train:

            fname = 'shapenet_training.dat'

            if Hans:

                fname = 'hans_training.dat'

        else:

            fname = 'shapenet_test.dat'
            if Hans:
                fname = 'hans_test.dat'

        with open(os.path.join(self.root, self.processed_folder, fname), 'wb') as f:

            torch.save(obj=names, f=f)

        return data, labels


class BalancedMNIST(MNIST):

    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'

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

            self.train_data, self.train_labels =generate_balanced_triplets(train_labels_class,train_data_class, mnist=True)

            #print(self.train_data.shape)

        else:

            test_data, test_labels = torch.load(os.path.join(mnist_set.root, mnist_set.processed_folder, mnist_set.test_file))

            test_labels_class, test_data_class= group_by_class(test_data, test_labels)

            self.test_data, self.test_labels = generate_balanced_triplets(test_labels_class, test_data_class, mnist=True)

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



def img_preproc(path_to_image, ResNet=True, ros=False):

    if not ros:
        img = cv2.imread(path_to_image)

    else:
        img = path_to_image

    if ResNet:
        W= 224
        H= 224
    else:
        W = 60
        H = 160

    x = cv2.resize(img, (W,H))
    x = np.asarray(x)
    #display_img(x)
    x = x.astype('float')
    #display_img(x)
    x = np.expand_dims(x, axis= 0)

    x = np.reshape(x, (x.shape[0], x.shape[3], x.shape[1], x.shape[2]))
    #display_img(x)

    return x/255.


def group_by_class(data, labels, classes=10, Hans =HANS):   #ids=None

    """
    Returns lists of len 10 grouping tensors
    belonging to the same digit/class at
    each indexed position

    """
    labels_class = []
    data_class = []
    #id_class = []

    if Hans:

        classes = 4

    # For each digit in the data
    for i in range(classes):
        # Check location of data labeled as current digit
        # and reduce to one-dimensional LongTensor

        indices = torch.squeeze((labels == i).nonzero())

        # Use produced indices to select rows in the original data tensors loaded
        # And add them to related list
        labels_class.append(torch.index_select(labels, 0, indices))
        data_class.append(torch.index_select(data, 0, indices))

        #if ids is not None:
        #    id_class.append(torch.index_select(ids, 0, indices))

    return labels_class, data_class


def generate_balanced_triplets(labels_class, data_class, mnist=False):

    data = []
    labels = []

    # Uncomment the following to check number of samples per class
    if mnist:

       min_ = 500
    else:

       min_ = min([x.shape[0] for x in labels_class])

    #print([x.shape[0] for x in labels_class])
    # Check here for different sample number
    for i in range(len(labels_class)):

        for j in range(min_):  # 500  # create 500*10 triplets

            if mnist:

                idx = random.randint(0, min_ - 1)

                data, labels = pick_samples(data_class, labels_class, data, labels, i, j, idx, min_)

            else:

                #Oversample   Fewer examples available
                r = [y for y in range(min_) if y != j]  # excluding j

                for idx in r:

                    data, labels = pick_samples(data_class, labels_class, data, labels, i, j, idx, min_)

    # print(torch.stack(data).shape)

    return torch.stack(data), torch.tensor(labels)


def pick_samples(data_class, labels_class, data, labels, i, j, idx, min_):

    # choose random class different from current one
    other_cls = [y for y in range(len(labels_class)) if y != i]

    rnd_cls = random.choice(other_cls)
    rnd_idx = random.randint(0, min_ - 1)

    data.append(torch.stack([data_class[i][j], data_class[i][idx], data_class[rnd_cls][rnd_idx]]))
    # data.append(torch.stack([data_class[i][j], data_class[i][rnd_dist], data_class[rnd_cls][j]]))

    # Append the pos neg labels for the two pairs
    labels.append([1, 0])

    return data, labels

