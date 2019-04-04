import torch
from embedding_extractor import path_embedding, array_embedding
import os
from segment import find_bboxes, convert_bboxes, crop_img
import numpy as np


def compute_similarity(qembedding, train_embeds):
    # Similarity matching against indexed data

    similarities = {}

    for emb_id, emb in train_embeds.items():
        # Cos sim reduces to dot product since embeddings are L2 normalized
        similarities[emb_id] = torch.mm(qembedding, emb.t()).item()

    # Return top-K results
    ranking = sorted(similarities.items(), key=lambda kv: kv[1], reverse=True)

    return ranking


def test(model, model_checkpoint, data_type, path_to_test, path_to_bags, device, trans, path_to_train_embeds, K=5, sthresh= 0.1):

    train_embeds = torch.load(path_to_train_embeds)

    # Code for test/inference time on one(few) shot(s)
    # for each query image

    if data_type == 'pickled':

        try:

            img_mat = np.load(path_to_bags)

        except Exception as e:

            print("Problem while reading provided input +\n")
            print(str(e))
            return

        all_imgs, timestamps = zip(*img_mat)

        for img in all_imgs:

            with open('pt_results/ranking_log.txt', 'w') as outr:

                #Segment using lightnet (YOLO)
                bboxes = find_bboxes(img, thr= sthresh)

                # Convert boxes back to original image resolution
                # And from YOLO format (center coord) to ROI format (top-left/bottom-right)
                n_bboxes = convert_bboxes(bboxes, img.shape)

                #Crop to each box found
                obj_list = crop_img(img, n_bboxes)

                for obj in obj_list:

                    qembedding = array_embedding(model, model_checkpoint, obj, \
                                                 device, transforms=trans)
                    ranking = compute_similarity(qembedding, train_embeds)

                    if K == 1:
                        label = ranking[0][0].split("_")[0]
                        print("The top most similar object is %s \n" % label)
                        outr("The top most similar object is %s \n" % label)

                    else:
                        outr.write("The %i most similar objects to the provided image are: \n" % K)

                        for key, val in ranking[:K - 1]:

                            label = key.split("_")[0]  # [:-1]

                            print(label + ": " + str(val) + "\n")
                            outr.write(label + ": " + str(val) + "\n")


        return None

    with open('pt_results/ranking_log.txt', 'w') as outr:

        class_wise_res = []

        for root, dirs, files in os.walk(path_to_test):

            if files:  #For each object class

                #tot_wrong = 0

                classname = str(root.split('/')[-1])

                outr.write(classname)

                for file in files: #For each example in that class

                    qembedding = path_embedding(model, model_checkpoint, os.path.join(root, file), \
                                                 device, transforms=trans)

                    ranking = compute_similarity(qembedding, train_embeds)

                    outr.write("The %i most similar objects to the provided image are: \n" % K)

                    correct_preds = 0

                    class_accs = []

                    if K == 1:

                        key, val = ranking[0]
                        label = key.split("_")[0]  # [:-1]
                        outr.write(label + ": " + str(val) + "\n")

                        if label == classname:

                            correct_preds += 1

                        else:

                            outr.write('{} mistaken for {}'.format(classname, label))
                            # tot_wrong += 1

                        class_accs.append(correct_preds)

                    else:
                        for key, val in ranking[:K - 1]:


                            label = key.split("_")[0] #[:-1]

                            outr.write(label + ": " + str(val) + "\n")

                            #Parse labels to binary as correct? Yes/No
                            if label == classname:

                                correct_preds += 1

                            else:

                                outr.write('{} mistaken for {}'.format(classname, label))
                                #tot_wrong += 1

                            avg_acc = correct_preds/K
                            class_accs.append(avg_acc)

                macro_avg = sum(class_accs)/len(class_accs)

                print('Mean average accuracy for class {} is {}'.format(classname, float(macro_avg)))
                outr.write('Mean average accuracy for class {} is {}'.format(classname, float(macro_avg)))
                class_wise_res.append((classname, macro_avg))

        return zip(*class_wise_res)