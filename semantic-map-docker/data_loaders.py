from torchvision.datasets import MNIST
import cv2
import os
from PIL import Image
import torch
import random
from siamese_models import ResSiamese

""""

Classes and methods to create balanced triplets
- In a custom case on own data set
- In the case of a standard set, i.e., extending the MNIST class


"""
HANS = False

KNOWN = ['chairs', 'bottles', 'papers', 'books', 'desks', 'boxes', 'windows', 'exit-signs', 'coat-racks', 'radiators']
NOVEL = ['fire-extinguishers', 'desktop-pcs', 'electric-heaters', 'lamps', 'power-cables', 'monitors', 'people', 'plants', 'bins', 'doors' ]
ALL= KNOWN+NOVEL

class_no =0
"""
class_dict = {}

for class_name in KNOWN:

    class_dict[class_name] = class_no
    class_no +=1
"""
n = 25

class BalancedTriplets(torch.utils.data.Dataset):

    global n

    """Generic class to load custom dataset and generate balanced triplets
       in the form <anchor, positive example, negative example>
       for both an image collection and a set of labels
    """
    #Local paths
    raw_folder = 'shapenet10'
    processed_folder = 'processed'
    training_file = 'shp10_training.pt'
    test_file = 'shp10_test.pt'
    to_val = 'val'


    def __init__(self, root, device, train=True, transform=None,  target_transform=None, N=10, ResNet=True, KNet=False):

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.resnet= ResNet
        self.knet = KNet
        self.KMidata = 'KMi_collection/'

        n = N


        if n == 20:
            #Use shapenet20 classes instead
            self.raw_folder = 'shapenet20'
            self.training_file = 'shp20_training.pt'
            self.test_file = 'shp20_test.pt'

        elif n == 25:
            #Use shapenet20 classes instead
            self.raw_folder = 'shapenet25'
            self.training_file = 'shp25_training.pt'
            self.test_file = 'shp25_test.pt'
            self.trainref_file = 'kmi25_training.pt'
            self.valref_file = 'kmi25_val.pt'


        if not os.path.isdir(os.path.join(self.root, self.processed_folder)):
            os.mkdir(os.path.join(self.root, self.processed_folder))

        self.prep(self.transform, self.train, resnet=self.resnet)

        if not self._check_exists(self.train):
            raise RuntimeError('Dataset not found.')

        if self.train:

            # Load explicitly from processed MNIST files created
            #train_data, train_labels = torch.load(
            #    os.path.join(self.root, self.processed_folder, self.training_file))

            #print(train_data.shape)

            # To then pass to new functions

            
            self.train_data, self.train_labels = generate_KNN_triplets(self.train, device, KNet= self.knet)

            """
            train_labels_class, train_data_class = group_by_class(train_data, train_labels, classes=10)
            #print(train_labels_class)

            self.train_data, self.train_labels = generate_balanced_triplets(train_labels_class, train_data_class)
            """
            print(self.train_data.shape)
            print(self.train_labels.shape)

        else:

            #test_data, test_labels = torch.load(
            #    os.path.join(self.root, self.processed_folder, self.test_file))

            self.test_data, self.test_labels = generate_KNN_triplets(self.train, device, KNet= self.knet)
            """
            test_labels_class, test_data_class = group_by_class(test_data, test_labels, classes=10)

            self.test_data, self.test_labels = generate_balanced_triplets(test_labels_class, test_data_class)

            """
            print(self.test_data.shape)
            print(self.test_labels.shape)

    def split_data(self, data_path, out_path, ratio=(.35, .35, .3)):

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
            """
            if self.transform is not None:
                img = self.transform(img)
            """
            img_ar.append(img)
        """
        if self.target_transform is not None:
            target = self.target_transform(target)
        """
        return img_ar, target

    def __len__(self):

        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self, train=True):

        if train:
            return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file))
        else:
            return os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))

    def prep(self, trans, train=True, resnet=True):

        print('Reading images and labels...')

        #Load from local
        if train:

            training_set = (self.read_files(os.path.join(self.root, self.raw_folder, 'train'), trans, ResNet=resnet ))
            # Save as .pt files

            ref_set = (self.read_files(os.path.join(self.root, self.KMidata, 'train'), trans, ResNet=resnet , KMI=True))

            with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as f:
                torch.save(training_set, f)

            with open(os.path.join(self.root, self.processed_folder, self.trainref_file), 'wb') as f:
                torch.save(ref_set, f)

        else:

            test_set = (self.read_files(os.path.join(self.root, self.raw_folder, self.to_val), trans, ResNet=resnet,train=False))

            ref_set = (self.read_files(os.path.join(self.root, self.KMidata, 'val'), trans, ResNet=resnet, train=False, KMI=True))

            with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:

                torch.save(test_set, f)

            with open(os.path.join(self.root, self.processed_folder, self.valref_file), 'wb') as f:
                torch.save(ref_set, f)

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


    def read_files(self, path, trans, train=True, ResNet=True, KMI=False):

        #discarded = []
        #kept=[]
        global n
        global class_list


        if train:

            total = 100

            if not KMI:

                total = 250

        elif (not train) and (n == 10):

            total = 82

        elif (not train) and (n == 20):

            total = 50

        elif (not train) and (n == 25):

                if KMI:
                    total = 25
                else:
                    total = 125

        if ResNet:

            data = torch.empty((total, 3, 224, 224))

        else:

            data = torch.empty((total, 3, 160, 60))


        labels = torch.empty((total))
        names = {}

        #Subfolders are named after classes here
        iteration = 0
        class_ = 0
        all_cs = []

        for root, dirs, files in os.walk(path):

            if files:

                classname = str(root.split('/')[-1])
                all_cs.append(classname)

                if n == 20 and (classname not in KNOWN):

                    #discarded.append(classname)
                    #Skip novel object classes
                    continue

                #kept.append(classname)
                #print(torch.LongTensor([class_]))
                #NOTE: the assumption is that images are grouped in subfolders by class
                #example_no = 1

                for file in files:


                    try:
                        filename = str(file.split('/')[-1])
                        img_tensor = img_preproc(os.path.join(root, file),trans)  # torch.from_numpy(img_preproc(os.path.join(root, file)))
                        data[iteration, :] = img_tensor

                    except Exception as e:

                        print(total)
                        print(str(e))
                        print(classname)
                        print(filename)

                    labels[iteration] = torch.LongTensor([class_])

                    # ID = <classname_filename>
                    names[classname+'_'+filename] = img_tensor
                    #example_no += 1
                    iteration += 1


                class_ += 1

        class_list = list(set(all_cs))

        #Save serialized object separately for IDs
        if train and KMI:
            fname = 'kmi_training.dat'

        elif train and not KMI:
            fname = 'training.dat'

        elif not train and KMI:
            fname = 'kmi_test.dat'

        else:
            fname = "test.dat"

        with open(os.path.join(self.root, self.processed_folder, fname), 'wb') as f:

            torch.save(obj=names, f=f)

        #print(set(discarded))
        #print(set(kept))

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

            self.train_data, self.train_labels = generate_balanced_triplets(train_labels_class,train_data_class, mnist=True)

            print(self.train_data.shape)

        else:

            test_data, test_labels = torch.load(os.path.join(mnist_set.root, mnist_set.processed_folder, mnist_set.test_file))

            test_labels_class, test_data_class = group_by_class(test_data, test_labels)

            self.test_data, self.test_labels = generate_balanced_triplets(test_labels_class, test_data_class, mnist=True)

            print(self.test_data.shape)


    def __getitem__(self, index):

        if self.train:

            imgs, target = self.train_data[index], self.train_labels[index]

        else:
            imgs, target = self.test_data[index], self.test_labels[index]

        img_ar = []

        for i in range(len(imgs)):

            #img = Image.fromarray(imgs[i].numpy(), mode='L')

            """
            if self.transform is not None:

                img = self.transform(img)

            """
            img_ar.append(i)

        """
        if self.target_transform is not None:

            target = self.target_transform(target)
        """
        return img_ar, target

    def __len__(self):

        if self.train:

            return len(self.train_data)

        else:

            return len(self.test_data)


def BGRtoRGB(img_array):
    img = img_array.copy()
    img[:, :, 0] = img_array[:, :, 2]
    img[:, :, 2] = img_array[:, :, 0]

    return img

def img_preproc(path_to_image, transform, ResNet=True, ros=False):


    if not ros:
        img = cv2.imread(path_to_image)
        img = BGRtoRGB(img)


    else:

        img = path_to_image

    """
    if ResNet:
        W= 224
        H= 224
    else:
        W = 60
        H = 160

    x = cv2.resize(img, (W,H))
    """
    #x = np.asarray(img) #x)

    x = Image.fromarray(img, mode='RGB')


    """
    #display_img(x)
    print(type(x))
    
    x = x.astype('float')/255.
    #display_img(x)
    x = np.expand_dims(x, axis= 0) #To include batch no later

    x = np.reshape(x, (x.shape[0], x.shape[3], x.shape[1], x.shape[2]))
    #display_img(x)
    """

    return transform(x)

import test

def generate_KNN_triplets(train, device, KNet=False):

    global n
    global class_list

    # Save serialized object separately for IDs
    if train:
        fname = 'training.dat'
        kfname = 'kmi_training.dat'

    else:
        fname = 'test.dat'
        kfname = 'kmi_test.dat'

    data_dict = torch.load(os.path.join('./data/processed', fname))
    kmi_dict = torch.load(os.path.join('./data/processed', kfname))

    # extract embeddings on all set given
    # pre-trained ResNet without retrain

    model = ResSiamese(feature_extraction=True).to(device)
    model.eval()

    embed_space = {}

    for imgkey, img in data_dict.items():  # i in range(data.size(0)):

        # img_tensor = data[i, :]
        img_tensor = img.view(1, img.shape[0], img.shape[1], img.shape[2]).to(device)
        embed_space[imgkey] = model.get_embedding(img_tensor)

    rgbd_embed_space = {}

    for imgkey, img in kmi_dict.items():  # i in range(data.size(0)):

        # img_tensor = data[i, :]
        img_tensor = img.view(1, img.shape[0], img.shape[1], img.shape[2]).to(device)
        rgbd_embed_space[imgkey] = model.get_embedding(img_tensor)

    triplet_data = []
    labels = []

    # CHANGED: anchors are KMi natural scenes now
    for imgkey, img in kmi_dict.items():
        # img_tensor = img.view(1, img.shape[0], img.shape[1], img.shape[2]).to(device)
        anchor_emb = rgbd_embed_space[imgkey]  # model.get_embedding(img_tensor)

        anchor_class = imgkey.split("_")[0]

        ranking = test.compute_similarity(anchor_emb, embed_space)

        # picking negative example from real imgs
        ranking_neg = test.compute_similarity(anchor_emb, rgbd_embed_space)

        positive_eg = None
        negative_eg = None

        # CHANGED: now not needed, two spaces are different
        for i in range(len(ranking)):
            # i.e., not starting from zero to exclude the embed itself

            # for i in range(1,len(ranking)):

            if positive_eg is not None:
                break

            key, val = ranking[i]

            top_match = data_dict[key]  # embed_space[key]

            top_class = key.split("_")[0]

            if top_class == anchor_class and positive_eg is None:
                positive_eg = top_match

        for j in range(1, len(ranking_neg)):

            if negative_eg is not None:
                break

            key, val = ranking_neg[j]

            top_match = kmi_dict[key]  # embed_space[key]

            top_class = key.split("_")[0]

            if top_class != anchor_class and negative_eg is None:
                negative_eg = top_match

        """
        #print("And Least similar example")
        key, val = ranking[1]
        positive_eg = data_dict[key]

        key, val = ranking[-1]
        negative_eg = data_dict[key] #embed_space[key]
        #print(key)
        #print(val)
        """
        triplet_data.append(torch.stack([img, positive_eg, negative_eg]))
        # print(torch.stack([img, positive_eg, negative_eg]).shape)

        # print(class_list)
        # import sys
        # sys.exit(0)
        if KNet:

            temp = torch.zeros(n)  # (len(KNOWN))
            j = class_list.index(anchor_class)
            temp[j] = 1  # Make a one-hot encoding of it

            labels.append(temp)

        else:

            labels.append([1, 0])

    if KNet:

        return torch.stack(triplet_data), torch.stack(labels)


    else:

        return torch.stack(triplet_data), torch.tensor(labels)



def group_by_class(data, labels, classes=10, Hans =HANS):   #ids=None

    """
    Returns lists of len 10 grouping tensors
    belonging to the same digit/class at
    each indexed position

    """
    labels_class = []
    data_class = []
    #id_class = []

    """
    if Hans:

        classes = 4
        
        
    """
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
                idx = random.choice(r)

                #for idx in r:

                data, labels = pick_samples(data_class, labels_class, data, labels, i, j, idx, min_)


    return torch.stack(data), torch.tensor(labels)


def pick_samples(data_class, labels_class, data, labels, i, j, idx, min_):

    # choose random class different from current one
    other_cls = [y for y in range(len(labels_class)) if y != i]

    rnd_cls = random.choice(other_cls)
    rnd_idx = random.randint(0, min_ - 1)

    data.append(torch.stack([data_class[i][j], data_class[i][idx], data_class[rnd_cls][rnd_idx]]))

    # Append the pos neg labels for the two pairs
    labels.append([1, 0])

    return data, labels

