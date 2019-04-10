import torch
from embedding_extractor import path_embedding, array_embedding
import os
from segment import segment
import numpy as np
from PIL import Image
import cv2

#Set of labels to retain from segmentation
keepers= ['person','chair','potted plant']

def compute_similarity(qembedding, train_embeds):
    # Similarity matching against indexed data

    similarities = {}

    for emb_id, emb in train_embeds.items():
        # Cos sim reduces to dot product since embeddings are L2 normalized
        similarities[emb_id] = torch.mm(qembedding, emb.t()).item()

    # Return top-K results
    ranking = sorted(similarities.items(), key=lambda kv: kv[1], reverse=True)

    return ranking


def test(model, model_checkpoint, data_type, path_to_test, path_to_bags, device, trans, path_to_train_embeds, K=5, sthresh= 1.0):

    train_embeds = torch.load(path_to_train_embeds)

    # Code for test/inference time on one(few) shot(s)
    # for each query image

    if data_type == 'pickled':

        try:

            img_mat = np.load(path_to_bags, encoding = 'latin1')

        except Exception as e:

            print("Problem while reading provided input +\n")
            print(str(e))
            return


        for img, timestamp in zip(*img_mat):

            #img2 = img.astype('float32')

            #Write temporary img file
            cv2.imwrite('temp.jpg', img)

            with open('pt_results/ranking_log.txt', 'w') as outr:

                yolo_preds = segment('temp.jpg', img)

                for idx, obj, yolo_label in enumerate(zip(*yolo_preds)):

                    #For each box wrapping an object
                    #Image.fromarray(obj, mode='RGB').show()
                    #Pre-process as usual validation/test images to extract embedding
                    qembedding = array_embedding(model, model_checkpoint, obj, \
                                                 device, transforms=trans)

                    if yolo_label in keepers:

                        #Add to collection of known objects for future rankings
                        #Reconcile format to labels used for other ShapeNet examples
                        if yolo_label == 'person':
                            label ='people'
                        elif yolo_label == 'chair':
                            label = 'chairs'
                        elif yolo_label == 'potted plant':
                            label = 'plants'

                        #Unique image identifier
                        img_id = label +'_'+str(idx)+ '_'+ str(timestamp)
                        train_embeds[img_id] = qembedding

                    else:
                        #Go ahead and classify by similarity
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


        #Save updated embeddings after YOLO segmentation
        with open(path_to_train_embeds, mode='wb') as outf:
            torch.save(obj=train_embeds, f=outf)

        print("Updated embeddings saved under %s" % path_to_train_embeds)

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