import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support
import numpy as np

def train(model, device, train_loader, epoch, optimizer, num_epochs, metric_avg):

    model.train()

    accurate_labels = 0
    all_labels = 0
    running_loss = 0

    # Requires early-stopping tool provided at https://github.com/Bjarten/early-stopping-pytorch

    labels = []
    predictions = []

    for batch_idx, (data, target) in enumerate(train_loader):

        # print(data[0].shape)
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

    # print(predictions[0].shape)
    # print(labels[0].shape)

    # Compute epoch-level metrics with sklearn
    """
    'micro': Calculate metrics globally 
    by counting the total true positives, 
    false negatives and false positives

    'macro': Calculate metrics for each label, 
    and find their unweighted mean. 
    This does not take label imbalance into account.

    """
    p, r, f1, sup = precision_recall_fscore_support(np.asarray(labels), np.asarray(predictions), average=metric_avg)

    print("Epoch {}/{}, Loss: {:.3f}, Accuracy: {:.3f}%".format(epoch + 1, num_epochs, epoch_loss, accuracy))
    # print(torch.Tensor([epoch_loss, accuracy]))

    return torch.Tensor([epoch_loss, accuracy, float(p), float(r)])