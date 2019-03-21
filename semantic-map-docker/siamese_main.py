import torch
from torch import optim
from torchvision import transforms
import os

#custom classes and methods
from plot_results import gen_plots
import data_loaders
from siamese_models import SimplerNet, NCCNet, ResSiamese
from pytorchtools import EarlyStopping
from embedding_extractor import save_embeddings
from train import train
from test import test

do_learn = True
feature_extraction = False
keep_embeddings = True
save_frequency = 2
batch_size = 16
lr = 0.0001
num_epochs = 1000
weight_decay = 0.0001
patience = 80
metric_avg = 'micro'
momentum = 0.9



def main(NCC=False, MNIST=True, ResNet=True):

    """
    Expects two command line arguments or flags
    first one specifing which model to use, e.g., simple or NCC
    the second one indicating the dataset, e.g., MNIST or SNS2

    """
    global num_epochs

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if ResNet:

        # Add hardcoded mean and variance values for torchvision pre-trained modules
        #means = [0.485, 0.456, 0.406]
        #stds = [0.229, 0.224, 0.225]
        means = (0.5,)
        stds = (1.0,)

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

    trans = transforms.Compose([transforms.Normalize(means, stds)])



    if NCC:

        model = NCCNet().to(device)

        params_to_update = model.parameters()  # all params
    else:

        model = ResSiamese(feature_extraction).to(device) #SimplerNet().to(device)


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


            test_loader = torch.utils.data.DataLoader(
                data_loaders.BalancedMNIST('./data', train=False, transform=mnist_trans, download=False), batch_size=batch_size,
                shuffle=False)

        else:


            train_loader = torch.utils.data.DataLoader(
               data_loaders.BalancedTriplets('./data', train=True, transform=trans), batch_size=batch_size,
               shuffle=True)

            test_loader = torch.utils.data.DataLoader(
                data_loaders.BalancedTriplets('./data', train=False, transform=trans), batch_size=batch_size,
                shuffle=False)

        optimizer = optim.SGD(params_to_update, lr=lr, momentum=momentum)
        #optimizer = optim.Adam(params_to_update, lr=lr, weight_decay=weight_decay)

        epoch_train_metrics = [] #torch.empty((1,2))#((num_epochs, 2))
        epoch_test_metrics = [] #torch.empty_like(epoch_train_metrics)

        for epoch in range(num_epochs):

            if early_stopping.early_stop:
                print("Early stopping")

                break

            epoch_train_metrics.append(train(model, device, train_loader, epoch, optimizer, num_epochs, metric_avg))

            test_m = test(model, device, test_loader, metric_avg)
            epoch_test_metrics.append(test_m)

            valid_loss = test_m[0]
            early_stopping(valid_loss, model)



        ## Plotting ##################################################################
        epoch_train_metrics = torch.stack(epoch_train_metrics, dim=0)
        epoch_test_metrics = torch.stack(epoch_test_metrics, dim=0)

        epoch_losses = torch.stack((epoch_train_metrics[:,0], epoch_test_metrics[:,0]), dim=1)
        epoch_accs = torch.stack((epoch_train_metrics[:,1], epoch_test_metrics[:,1]), dim=1)

        epoch_ps = torch.stack((epoch_train_metrics[:,2], epoch_test_metrics[:,2]), dim=1)
        epoch_rs = torch.stack((epoch_train_metrics[:,3], epoch_test_metrics[:,3]), dim=1)

        gen_plots(epoch_losses, epoch_accs, num_epochs, MNIST, NCC)

        #Gen precision and recall plots
        gen_plots(epoch_ps, epoch_rs, num_epochs, MNIST, NCC, precrec=True)


    else:

        #Code for test/inference time
        pass

    if keep_embeddings:

        #Warning: available for custom set only, no MNIST
        save_embeddings(model, 'pt_results/embed_checkpoint.pt', './data/processed/shapenet_training.dat', \
                        device, transforms=trans)


if __name__ == '__main__':


    print("Running siamese net on SNS2")
    main(NCC=False, MNIST=False)

    """
    Reproduces old runs 
    # print("Running siamese net on MNIST")
    # main(NCC=False, MNIST=True)
    #print("Running NCC siamese net on MNIST")
    #main(NCC=True, MNIST=True)
    #print("Running NCC siamese net on SNS2")
    #main(NCC=True, MNIST=False)
    """