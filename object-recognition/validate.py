import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support, accuracy_score,roc_auc_score
import numpy as np
from models import TripletLoss

def validate(model, device, test_loader, metric_avg, knet=False,nnet=False):

    model.eval()
    criterion = TripletLoss()

    with torch.no_grad():

        accurate_labels = 0
        all_labels = 0
        running_loss = 0

        labels=[]
        predictions=[]
        pos_proba = []

        for batch_idx, (data, target) in enumerate(test_loader):

            for i in range(len(data)):
                data[i] = data[i].to(device)

            target = target.type(torch.LongTensor).to(device)

            if nnet:

                # No classification component involved
                emb_a, emb_p, emb_n = model(data)
                classif_loss = 0.0

            elif knet:
                target = torch.max(target, 1)[1]

                emb_a, emb_p, emb_n, output_logits = model(data)
                classif_loss = F.cross_entropy(output_logits, target)

            else:

                # base resnet with binary classifier
                output_positive, emb_a, emb_p = model(data[:2])
                output_negative, _, emb_n = model(data[0:3:2])

                target_positive = torch.squeeze(target[:, 0])
                target_negative = torch.squeeze(target[:, 1])

                loss_positive = F.cross_entropy(output_positive, target_positive)
                loss_negative = F.cross_entropy(output_negative, target_negative)

                classif_loss = loss_positive + loss_negative

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

            triplet_loss = criterion(emb_a, emb_p, emb_n)
            loss = classif_loss + triplet_loss

            if nnet:

                norm_tloss = emb_a.shape[0] * triplet_loss.item()
                running_loss += norm_tloss
                accurate_labels = 0
                all_labels = all_labels + data[0].shape[0] # Cannot really classify in this case

            elif knet:
                norm_tloss = emb_a.shape[0] * triplet_loss.item()
                norm_closs = output_logits.shape[0] * classif_loss.item()
                running_loss += norm_tloss + norm_closs
                # Multi-class instead of binary
                predictions.extend(torch.argmax(output_logits, dim=1).tolist())
                labels.extend(target.tolist())
                #accurate_labels = torch.sum(torch.argmax(output_logits, dim=1) == target).cpu()
                all_labels = all_labels + len(target)
                metric_avg = 'weighted'  # for computing the accuracy

        accuracy = accuracy_score(labels, predictions)

        #accuracy = 100. * accurate_labels / all_labels
        epoch_loss = running_loss/all_labels

        if nnet:
            p = 0.0
            r = p
            f1 = p
            sup = 0
        else:
            p, r, f1, sup = precision_recall_fscore_support(np.asarray(labels), np.asarray(predictions), average=metric_avg)

        roc_auc = 0.0 #roc_auc_score(np.asarray(labels), np.asarray(pos_proba))

        print('Val accuracy: {}/{} ({:.6f}%)\t Loss: {:.6f}, Precision: {:.6f}, Recall: {:.6f}, F1: {:.6f}, ROC_AUC: {:.6f}'.format(int(accurate_labels), int(all_labels), accuracy, epoch_loss, p, r, f1, roc_auc))

        return torch.Tensor([epoch_loss, accuracy, float(p), float(r), float(roc_auc)])
