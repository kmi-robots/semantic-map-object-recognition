import torch
from torch import optim
from torchvision import transforms
import os
import argparse

#custom classes and methods
from plot_results import gen_plots, bar_plot
import data_loaders
from siamese_models import NCCNet, ResSiamese, ResTwoBranch, KNet, NNet, ImprintedKNet #, ContrastiveLoss
from pytorchtools import EarlyStopping
from embedding_extractor import extract_embeddings
from train import train
from validate import validate
from test import test
from imprint import imprint

#These parameters can be tweaked---------------------------------------------------------#
do_learn = False
feature_extraction = False
keep_embeddings = True

KNET = True
NNET = False #True
imprinting = False
two_branch=False

save_frequency = 2
batch_size = 16
lr = 0.0001
num_epochs = 1000
weight_decay = 0.00001
patience = 20
metric_avg = 'binary'
momentum = 0.9
segmentation_threshold = 0.01

N = 25 #20 #10 #No of object classes
path_to_query_data = './data/shapenet20/test/' #fire-extinguishers/9. fire-extinguisher.jpg
path_to_train_embeds = './pt_results/paper/embeddings_absL1moved.dat'
K = 1 #

path_to_bags = './data/KMi_collection/test/tagged_KMi.json' #'robot_collected.npy'

STN = False #Whether to use Spatial Transformer module on input or not

if STN:
    #Decrease learning rate
    lr = 0.00001
#-----------------------------------------------------------------------------------------#


#Hardcoded variables -------------------------------------------------------------------------#
model_checkpoint = 'pt_results/paper/checkpoint_absL1moved.pt' #hardcoded in pytorchtools.py
path_to_train_data ='./data/processed/training.dat' #hardcoded in data_loaders.py ln. 205
#--- Simpler not to change -----------------------------------------------------------------------#


def main(input_type, NCC=False, MNIST=True, ResNet=True):

    """
    Expects two command line arguments or flags
    first one specifing which model to use, e.g., simple or NCC
    the second one indicating the dataset, e.g., MNIST or SNS2

    """
    global num_epochs, keep_embeddings

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if ResNet:

        # Add hardcoded mean and variance values for torchvision pre-trained modules
        means = [0.485, 0.456, 0.406]
        stds = [0.229, 0.224, 0.225]

        mnist_trans = transforms.Compose([transforms.Resize((224, 224)),
                                          transforms.Grayscale(3),
                                          transforms.ToTensor(),
                                          transforms.Normalize(means, stds)])
    else:

        #simply standardize
        means = (0.5,)
        stds = (1.0,)
        mnist_trans = transforms.Compose([transforms.Resize((160, 60)),
                                    transforms.Grayscale(3),
                                    transforms.ToTensor(),
                                    transforms.Normalize(means, stds)])

    if NCC:

        #Supported in CPU-only for now
        device = torch.device('cpu')

        model = NCCNet().to(device)
        params_to_update = model.parameters()  # all params

        # Transformations applied to the images on training
        trans_train = transforms.Compose([
            transforms.Resize((160, 60)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(means, stds)])

        # Transformations applied to the images on validation and at test time
        trans_val = transforms.Compose([
            transforms.Resize((160, 60)),
            transforms.ToTensor(),
            transforms.Normalize(means, stds)])

    else:

        # Transformations applied to the images on training
        trans_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(means, stds)])

        # Transformations applied to the images on validation and at test time
        trans_val = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(means, stds)])

        """
        if STN:

            # Transformations applied to the images on training
            trans_train = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(means, stds)])

            # Transformations applied to the images on validation and at test time
            trans_val = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(means, stds)])

        """

        if two_branch:

            model = ResTwoBranch(feature_extraction=feature_extraction, stn=STN).to(device)


        elif NNET:

            model = NNet(feature_extraction=feature_extraction).to(device)

        elif KNET and not imprinting:

            model = KNet(feature_extraction=feature_extraction, num_classes=N).to(device)

        elif imprinting:
            model = ImprintedKNet(feature_extraction=feature_extraction, num_classes=N).to(device)


        else:

            model = ResSiamese(feature_extraction=feature_extraction, stn=STN).to(device)

        if feature_extraction:

            params_to_update = [param for param in model.parameters() if param.requires_grad]
            #only the last layers when doing feature extraction

        else:

            params_to_update = model.parameters()  # all params

    base_trans = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(means, stds)])

    if not os.path.isdir('./data'):

        os.mkdir('data')


    if not os.path.isdir('./pt_results'):

        os.mkdir('pt_results')

    if not os.path.isdir('./pt_results/paper'):

        os.mkdir('pt_results/paper')


    early_stopping = EarlyStopping(patience=patience, verbose=True)

    if do_learn:  # training mode

        if MNIST:

            train_loader = torch.utils.data.DataLoader(
              data_loaders.BalancedMNIST('./data', train=True, transform=mnist_trans, download=True), batch_size=batch_size,
               shuffle=True)


            val_loader = torch.utils.data.DataLoader(
                data_loaders.BalancedMNIST('./data', train=False, transform=mnist_trans, download=False), batch_size=batch_size,
                shuffle=False)


        else:


            train_loader = torch.utils.data.DataLoader(
               data_loaders.BalancedTriplets('./data', device, train=True, N=N, transform=base_trans, ResNet=ResNet, KNet=KNET), batch_size=batch_size,
               shuffle=True)

            val_loader = torch.utils.data.DataLoader(
                data_loaders.BalancedTriplets('./data', device, train=False, N=N, transform=base_trans, ResNet=ResNet, KNet=KNET), batch_size=batch_size,
                shuffle=False)

        optimizer = optim.SGD(params_to_update, lr=lr, momentum=momentum, weight_decay=weight_decay )
        #optimizer = optim.Adam(params_to_update, lr=lr, weight_decay=weight_decay)

        epoch_train_metrics = [] #torch.empty((1,2))#((num_epochs, 2))
        epoch_val_metrics = [] #torch.empty_like(epoch_train_metrics)

        #Imprint weight of the model classifier first
        if imprinting:
            imprint(model, device, train_loader, num_classes=N)

        for epoch in range(num_epochs):

            if early_stopping.early_stop:
                print("Early stopping")

                break

            epoch_train_metrics.append(train(model, device, train_loader, epoch, optimizer, num_epochs, metric_avg, knet=KNET,nnet=NNET))

            val_m = validate(model, device, val_loader, metric_avg, knet=KNET,nnet=NNET)

            epoch_val_metrics.append(val_m)

            valid_loss = val_m[0]
            early_stopping(valid_loss, model)

        ## Plotting ##------------------------------------------------------------------#
        epoch_train_metrics = torch.stack(epoch_train_metrics, dim=0)
        epoch_val_metrics = torch.stack(epoch_val_metrics, dim=0)

        epoch_losses = torch.stack((epoch_train_metrics[:,0], epoch_val_metrics[:,0]), dim=1)
        epoch_accs = torch.stack((epoch_train_metrics[:,1], epoch_val_metrics[:,1]), dim=1)

        epoch_ps = torch.stack((epoch_train_metrics[:,2], epoch_val_metrics[:,2]), dim=1)
        epoch_rs = torch.stack((epoch_train_metrics[:,3], epoch_val_metrics[:,3]), dim=1)

        epoch_roc_auc = torch.stack((epoch_train_metrics[:,4], epoch_val_metrics[:,4]), dim=1)
        gen_plots(epoch_losses, epoch_accs, num_epochs, MNIST, NCC)

        #Gen precision and recall plots
        gen_plots(epoch_ps, epoch_rs, num_epochs, MNIST, NCC, precrec=True)
        #-------------------------------------------------------------------------------#

        #Gen ROC AUC score plot
        gen_plots(epoch_roc_auc, epoch_roc_auc, num_epochs, MNIST, NCC, rocauc=True)
        #-------------------------------------------------------------------------------#

    else:

        #Test on held-out set


        class_wise_res = test(model, model_checkpoint, input_type, path_to_query_data, path_to_bags,\
                              device, base_trans, path_to_train_embeds, K, N, sthresh=segmentation_threshold)

        #Test plot grouped by class
        if class_wise_res is not None:
            #Evaluation only available for data held out from ground truth
            #When evaluating on robot-collected data, rankings are logged anyways
            bar_plot(class_wise_res)

        keep_embeddings = False


    if keep_embeddings:

        #Warning: available for custom set only, no MNIST
        extract_embeddings(model, model_checkpoint, path_to_train_data, \
                        device, path_to_train_embeds, transforms=base_trans)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('it', help='Input type at inference time: can be one between reference, pickled or json (if environment data have been tagged)')
    args = parser.parse_args()

    main(args.it, NCC=False, MNIST=False, ResNet=True)

    """
    Reproduces old runs 
    # print("Running siamese net on MNIST")
    # main(NCC=False, MNIST=True)
    #print("Running NCC siamese net on MNIST")
    #main(NCC=True, MNIST=True)
    #print("Running NCC siamese net on SNS2")
    #main(NCC=True, MNIST=False)
    """