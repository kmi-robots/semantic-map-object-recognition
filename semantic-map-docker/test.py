import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support


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
