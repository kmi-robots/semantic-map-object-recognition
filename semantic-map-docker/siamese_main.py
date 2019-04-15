import torch
from torch import optim
from torchvision import transforms
import os
import argparse

#custom classes and methods
from plot_results import gen_plots, bar_plot
import data_loaders
from siamese_models import SimplerNet, NCCNet, ResSiamese #, ContrastiveLoss
from pytorchtools import EarlyStopping
from embedding_extractor import extract_embeddings
from train import train
from validate import validate
from test import test


#These parameters can be tweaked---------------------------------------------------------#
do_learn = False
feature_extraction = False
keep_embeddings = True
save_frequency = 2
batch_size = 16
lr = 0.0001
num_epochs = 1000
weight_decay = 0.0001
patience = 20
metric_avg = 'binary'
momentum = 0.9
segmentation_threshold = 0.1

N = 20 #No of object classes
path_to_query_data = './data/all-objects/test/' #fire-extinguishers/9. fire-extinguisher.jpg
path_to_train_embeds = './pt_results/embeddings.dat'
K = 5
path_to_bags ='./data/robot_collected.npy'
STN = False #Whether to use Spatial Transformer module on input or not
#-----------------------------------------------------------------------------------------#


#Hardcoded variables -------------------------------------------------------------------------#
model_checkpoint = 'pt_results/checkpoint.pt' #hardcoded in pytorchtools.py
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
        #means = (0.5,)
        #stds = (0.5,)
        #stds = (1.0,)

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

    #Transformations applied to the images on training
    trans_train = transforms.Compose([
        transforms.RandomResizedCrop((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(means, stds)])

    # Transformations applied to the images on validation and at test time
    trans_val = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(means, stds)])

    if NCC:

        model = NCCNet().to(device)

        params_to_update = model.parameters()  # all params
    else:

        model = ResSiamese(feature_extraction, stn=STN).to(device) #SimplerNet().to(device)


        if feature_extraction:

            params_to_update = [param for param in model.parameters() if param.requires_grad]
            #only the last layers when doing feature extraction

        else:

            params_to_update = model.parameters()  # all params



    if not os.path.isdir('./data'):

        os.mkdir('data')

    if not os.path.isdir('./data/embeddings'):
            os.mkdir('data/embeddings')

    if not os.path.isdir('./pt_results'):

        os.mkdir('pt_results')

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
               data_loaders.BalancedTriplets('./data', train=True, N=N, transform=trans_train), batch_size=batch_size,
               shuffle=True)

            val_loader = torch.utils.data.DataLoader(
                data_loaders.BalancedTriplets('./data', train=False, N=N, transform=trans_val), batch_size=batch_size,
                shuffle=False)

        optimizer = optim.SGD(params_to_update, lr=lr, momentum=momentum)
        #optimizer = optim.Adam(params_to_update, lr=lr, weight_decay=weight_decay)

        epoch_train_metrics = [] #torch.empty((1,2))#((num_epochs, 2))
        epoch_val_metrics = [] #torch.empty_like(epoch_train_metrics)

        for epoch in range(num_epochs):

            if early_stopping.early_stop:
                print("Early stopping")

                break

            epoch_train_metrics.append(train(model, device, train_loader, epoch, optimizer, num_epochs, metric_avg))


            val_m = validate(model, device, val_loader, metric_avg)
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
                              device, trans_val, path_to_train_embeds, K, segmentation_threshold)

        #Test plot grouped by class
        if class_wise_res is not None:
            #Evaluation only available for data held out from ground truth
            #When evaluating on robot-collected data, rankings are logged anyways
            bar_plot(class_wise_res)

        keep_embeddings = False


    if keep_embeddings:


        #Warning: available for custom set only, no MNIST
        extract_embeddings(model, model_checkpoint, path_to_train_data, \
                        device, path_to_train_embeds, transforms=trans_val)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('it', help='Input type at inference time: can be one between reference or pickled')
    args = parser.parse_args()

    main(args.it, NCC=False, MNIST=False)

    """
    Reproduces old runs 
    # print("Running siamese net on MNIST")
    # main(NCC=False, MNIST=True)
    #print("Running NCC siamese net on MNIST")
    #main(NCC=True, MNIST=True)
    #print("Running NCC siamese net on SNS2")
    #main(NCC=True, MNIST=False)
    """