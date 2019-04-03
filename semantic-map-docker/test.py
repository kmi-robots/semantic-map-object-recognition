import torch
#import torch.nn.functional as F
#import numpy as np
#from sklearn.metrics import average_precision_score
from embedding_extractor import query_embedding
import os


def test(model, model_checkpoint, path_to_test, device, trans, path_to_train_embeds, K=5):

    # Code for test/inference time on one(few) shot(s)
    # for each query image
    train_embeds = torch.load(path_to_train_embeds)
    class_wise_res = []
    class_ratios = []

    for root, dirs, files in os.walk(path_to_test):

        if files:  #For each object class

            #tot_wrong = 0

            classname = str(root.split('/')[-1])

            for file in sample: #For each example in that class

                qembedding = query_embedding(model, model_checkpoint, path_to_query_data, \
                                             device, transforms=trans)

                # Similarity matching against indexed data

                similarities = {}

                for emb_id, emb in train_embeds.items():

                    # Cos sim reduces to dot product since embeddings are L2 normalized
                    similarities[emb_id] = torch.mm(qembedding, emb.t()).item()

                # Return top-K results
                ranking = sorted(similarities.items(), key=lambda kv: kv[1], reverse=True)

                print("The %i most similar objects to the provided image are: \n" % K)

                correct_preds = 0

                class_accs = []

                for key, val in ranking[:K - 1]:

                    label = key.split("_")[0][:-1]
                    print(label + ": " + str(val) + "\n")

                    #Parse labels to binary as correct? Yes/No
                    if label == classname:

                        correct_preds += 1

                    else:

                        print('%s mistaken for %s'.format(classname, label))
                        #tot_wrong += 1

                    avg_acc = correct_preds/K
                    class_accs.extend(avg_acc)

            macro_avg = sum(class_accs)/len(class_accs)

            print('Mean average accuracy for class %s is %f'.format(classname, float(macro_avg)))
            class_wise_res.append((classname, macro_avg))

    return zip(*class_wise_res)