import torch
from torch import optim
import torch.nn.functional as F
from torchvision import transforms
import os
from sklearn.metrics import precision_recall_fscore_support
import numpy as np

#custom classes and methods
from plot_results import gen_plots
import data_loaders
from siamese_models import SimplerNet, NCCNet, ResSiamese
from pytorchtools import EarlyStopping


do_learn = True
save_frequency = 2
batch_size = 16
lr = 0.00005
num_epochs = 300
weight_decay = 0.0001
patience = 80
metric_avg = 'micro'
feature_extraction = False
momentum = 0.9



def train(model, device, train_loader, epoch, optimizer):

    model.train()

    accurate_labels = 0
    all_labels = 0
    running_loss = 0

    #Requires early-stopping tool provided at https://github.com/Bjarten/early-stopping-pytorch

    labels = []
    predictions = []

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

        loss_positive = F.cross_entropy(output_positive, target_positive)
        loss_negative = F.cross_entropy(output_negative, target_negative)

        loss = loss_positive + loss_negative

        loss.backward()

        optimizer.step()

        norm_loss_p = output_positive.shape[0] * loss_positive.item()
        norm_loss_n = output_negative.shape[0] * loss_negative.item()

        running_loss += norm_loss_p + norm_loss_n


        accurate_labels_positive = torch.sum(torch.argmax(output_positive, dim=1) == target_positive).cpu()
        accurate_labels_negative = torch.sum(torch.argmax(output_negative, dim=1) == target_negative).cpu()

        # To then calculate epoch-level precision recall F1
        predictions.extend(torch.argmax(output_positive, dim=1).tolist())
        predictions.extend(torch.argmax(output_negative, dim=1).tolist())
        labels.extend(target_positive.tolist())
        labels.extend(target_negative.tolist())

        accurate_labels = accurate_labels + accurate_labels_positive + accurate_labels_negative
        all_labels = all_labels + len(target_positive) + len(target_negative)

    accuracy = 100 * accurate_labels / all_labels
    epoch_loss = running_loss / all_labels

    #print(predictions[0].shape)
    #print(labels[0].shape)

    #Compute epoch-level metrics with sklearn
    """
    'micro': Calculate metrics globally 
    by counting the total true positives, 
    false negatives and false positives
    
    'macro': Calculate metrics for each label, 
    and find their unweighted mean. 
    This does not take label imbalance into account.
    
    """
    p, r, f1, sup = precision_recall_fscore_support(np.asarray(labels), np.asarray(predictions), average=metric_avg)


    print("Epoch {}/{}, Loss: {:.3f}, Accuracy: {:.3f}%".format(epoch+1,num_epochs, epoch_loss,accuracy))
    #print(torch.Tensor([epoch_loss, accuracy]))

    return torch.Tensor([epoch_loss, accuracy, float(p), float(r)])



def test(model, device, test_loader):
    model.eval()

    with torch.no_grad():
        accurate_labels = 0
        all_labels = 0
        running_loss = 0

        labels=[]
        predictions=[]

        for batch_idx, (data, target) in enumerate(test_loader):


            for i in range(len(data)):
                data[i] = data[i].to(device)

            output_positive = model(data[:2])
            output_negative = model(data[0:3:2])

            target = target.type(torch.LongTensor).to(device)
            target_positive = torch.squeeze(target[:, 0])
            target_negative = torch.squeeze(target[:, 1])

            loss_positive = F.cross_entropy(output_positive, target_positive) #target of 1
            loss_negative = F.cross_entropy(output_negative, target_negative) #target of 0

            #loss = loss_positive + loss_negative

            norm_loss_p = output_positive.shape[0] * loss_positive.item() #as cross_entropy loss is the mean over batch size by default
            norm_loss_n = output_negative.shape[0] * loss_negative.item() #as cross_entropy  loss is the mean over batch size by default

            running_loss += norm_loss_p + norm_loss_n

            accurate_labels_positive = torch.sum(torch.argmax(output_positive, dim=1) == target_positive).cpu()
            accurate_labels_negative = torch.sum(torch.argmax(output_negative, dim=1) == target_negative).cpu()

            # To then calculate epoch-level precision recall F1
            predictions.extend(torch.argmax(output_positive, dim=1).tolist())
            predictions.extend(torch.argmax(output_negative, dim=1).tolist())
            labels.extend(target_positive.tolist())
            labels.extend(target_negative.tolist())


            accurate_labels = accurate_labels + accurate_labels_positive + accurate_labels_negative
            all_labels = all_labels + len(target_positive) + len(target_negative)

        accuracy = 100. * accurate_labels / all_labels
        epoch_loss = running_loss/all_labels
        p, r, f1, sup = precision_recall_fscore_support(np.asarray(labels), np.asarray(predictions), average=metric_avg)

        print('Test accuracy: {}/{} ({:.3f}%)\t Loss: {:.6f}'.format(accurate_labels, all_labels, accuracy, epoch_loss))


        return torch.Tensor([epoch_loss, accuracy,float(p), float(r)])


def oneshot(model, device, data):

    model.eval()

    with torch.no_grad():
        for i in range(len(data)):
            data[i] = data[i].to(device)

        output = model(data)
        return torch.squeeze(torch.argmax(output, dim=1)).cpu().item()


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

    trans = transforms.Compose([ transforms.Normalize(means, stds)])



    if NCC:

        model = NCCNet().to(device)

        params_to_update = model.parameters()  # all params
    else:

        model = ResSiamese(feature_extraction).to(device) #SimplerNet().to(device)

        if feature_extraction:

            params_to_update= [param for param in model.parameters() if param.requires_grad]

            #only the last layers when doing feature extraction

        else:

            params_to_update = model.parameters()  # all params



    if not os.path.isdir('./data'):

        os.mkdir('data')

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

            num_epochs = 1000

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

            epoch_train_metrics.append(train(model, device, train_loader, epoch, optimizer))

            #torch.stack(
            #(epoch_train_metrics, train(model, device, train_loader, epoch, optimizer)), dim=1)

            test_m = test(model, device, test_loader)
            epoch_test_metrics.append(test_m)
            #torch.stack((epoch_test_metrics, test_m),
            #                             dim=0)

            # epoch_train_metrics[epoch, :] = train(model, device, train_loader, epoch, optimizer)
            valid_loss = test_m[0]
            early_stopping(valid_loss, model)



            if epoch & save_frequency == 0:

                if MNIST & NCC:

                    torch.save(model, 'pt_results/MNIST/siamese_{:03}_NCC.pt'.format(epoch))

                elif MNIST & (not NCC):

                    torch.save(model, 'pt_results/MNIST/siamese_{:03}.pt'.format(epoch))

                elif (not MNIST) & NCC:

                    torch.save(model, 'pt_results/SNS2/siamese_{:03}_NCC.pt'.format(epoch))

                else:

                    torch.save(model, 'pt_results/SNS2/siamese_{:03}.pt'.format(epoch))

        epoch_train_metrics = torch.stack(epoch_train_metrics, dim=0)
        epoch_test_metrics = torch.stack(epoch_test_metrics, dim=0)

        epoch_losses = torch.stack((epoch_train_metrics[:,0], epoch_test_metrics[:,0]), dim=1)
        epoch_accs = torch.stack((epoch_train_metrics[:,1], epoch_test_metrics[:,1]), dim=1)

        epoch_ps = torch.stack((epoch_train_metrics[:,2], epoch_test_metrics[:,2]), dim=1)
        epoch_rs = torch.stack((epoch_train_metrics[:,3], epoch_test_metrics[:,3]), dim=1)

        gen_plots(epoch_losses, epoch_accs, num_epochs, MNIST, NCC)

        #Gen precision and recall plots
        gen_plots(epoch_ps, epoch_rs, num_epochs, MNIST, NCC, precrec=True)

    else:  # prediction
        prediction_loader = torch.utils.data.DataLoader(
            BalancedTriplets('./data', train=False, transform=trans), batch_size=batch_size,
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

    #print("Running simple net on MNIST")
    #main(NCC=False, MNIST=True)
    print("Running simple net on SNS2")
    main(NCC=False, MNIST=False)
    #print("Running NCC net on MNIST")
    #main(NCC=True, MNIST=True)
    #print("Running NCC net on SNS2")
    #main(NCC=True, MNIST=False)
