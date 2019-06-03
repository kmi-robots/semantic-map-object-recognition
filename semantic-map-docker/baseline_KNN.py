""""
Very similar to test.py but here embeddings are simply
extracted from a pre-trained ResNet without siamese
re-train/fine-tuning

"""
import torch
import embedding_extractor as emb_ext
import test
import os
from sklearn.metrics import classification_report, accuracy_score
from collections import Counter

old_setup = False
known_only = True
siamese_boosted = True

def run_baseline(device, transforms,  K, N, n_support, KNOWN, path_to_embeds):

    #Derive support set
    if N == 10 and old_setup:

        #10_classes for original shapenet comparison

        path_to_test = './data/shapenet10/train'

        path_to_support = './data/shapenet10/val'
        K=1  #Only interested in top-match


    elif N == 10 and not old_setup:

        path_to_test = './data/shapenet10/val'

        path_to_support = './data/shapenet10/train'


    elif N == 20 and known_only:

        path_to_test = './data/shapenet20/val'
        path_to_support = './data/shapenet20/train'


    else:

        path_to_test = './data/shapenet20/test'

        if n_support == 5:

            path_to_support = path_to_support = path_to_test + "/../val"

        elif n_support == 10:

            path_to_support = path_to_support = path_to_test + "/../train"


    if siamese_boosted:

        support_embeds = torch.load(path_to_embeds)

    else:

        support_embeds = {}
        #Create embedding space for support set
        for root, dirs, files in os.walk(path_to_support):

            if files:

                classname = str(root.split('/')[-1])

                if N == 20 and known_only and classname not in KNOWN:

                    continue

                for file in files:

                    filename = str(file.split('/')[-1])
                    support_embeds[classname+'_'+filename] = emb_ext.base_embedding(os.path.join(root, file), device, transforms=transforms)

    print(len(support_embeds.keys()))


    #Map test/query embeddings to the same space
    with open('pt_results/baseline_ranking_log.txt', 'w') as outr:

        class_wise_res = []
        y_true = []
        y_pred = []
        kn_true = []
        kn_pred = []
        nw_true = []
        nw_pred = []

        # Then make predictions for all test examples (Known + Novel)
        for root, dirs, files in os.walk(path_to_test):

            if files:  # For each object class

                # tot_wrong = 0

                classname = str(root.split('/')[-1])

                outr.write(classname)
                # all_classes.append(classname)

                if N == 20 and known_only and classname not in KNOWN:
                    continue

                for file in files:  # For each example in that class

                    y_true.append(classname)

                    if classname in KNOWN:

                        kn_true.append(classname)

                    else:

                        nw_true.append(classname)

                    print("%-----------------------------------------------------------------------% \n")
                    print("Looking at file %s \n" % file)
                    outr.write("Looking at file %s \n" % file)

                    qembedding = emb_ext.base_embedding(os.path.join(root, file), device, transforms=transforms)

                    ranking = test.compute_similarity(qembedding, support_embeds)

                    print("The %i most similar objects to the provided image are: \n" % K)
                    outr.write("The %i most similar objects to the provided image are: \n" % K)

                    correct_preds = 0

                    class_accs = []

                    if K == 1:

                        key, val = ranking[0]
                        label = key.split("_")[0]  # [:-1]

                        print(label + ": " + str(val) + "\n")
                        outr.write(label + ": " + str(val) + "\n")
                        print("With unique ID %s \n" % key)
                        outr.write("With unique ID %s \n" % key)

                        if label == classname:

                            correct_preds += 1

                        else:

                            print('{} mistaken for {}'.format(classname, label))
                            outr.write('{} mistaken for {}'.format(classname, label))
                            print("With unique ID %s \n" % key)
                            outr.write("With unique ID %s \n" % key)
                            # tot_wrong += 1

                        class_accs.append(correct_preds)


                    else:

                        votes = Counter()
                        ids = {}

                        # Majority voting with discounts by distance from top position
                        for k, (key, val) in enumerate(ranking[:K - 1]):
                            label = key.split("_")[0]  # [:-1]

                            votes[label] += 1 / (k + 1)

                            ids[label] = key

                            print(label + ": " + str(val) + "\n")
                            outr.write(label + ": " + str(val) + "\n")
                            print("With unique ID %s \n" % key)
                            outr.write("With unique ID %s \n" % key)

                        win_label, win_score = max(votes.items(), key=lambda x: x[1])
                        win_id = ids[win_label]

                        if win_label == classname:
                            correct_preds += 1

                        else:

                            print('{} mistaken for {}'.format(classname, win_label))
                            outr.write('{} mistaken for {}'.format(classname, win_label))
                            print("With unique ID %s \n" % win_id)
                            outr.write("With unique ID %s \n" % win_id)
                            # tot_wrong += 1

                        # avg_acc = correct_preds/K
                        class_accs.append(correct_preds)
                        label = win_label

                    y_pred.append(label)

                    if classname in KNOWN:

                        kn_pred.append(label)

                    else:

                        nw_pred.append(label)

                    print("%EOF---------------------------------------------------------------------% \n")

        print("Class-wise test results \n")
        print(classification_report(y_true, y_pred))  # , target_names=all_classes))
        print(accuracy_score(y_true, y_pred))

        print("Known objects test results \n")
        print(classification_report(kn_true, kn_pred))  # , target_names=all_classes))
        print(accuracy_score(kn_true, kn_pred))

        print("Novel objects test results \n")
        print(classification_report(nw_true, nw_pred))  # , target_names=all_classes))
        print(accuracy_score(nw_true, nw_pred))