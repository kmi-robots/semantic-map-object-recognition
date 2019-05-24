import torch
from embedding_extractor import path_embedding, array_embedding, base_embedding
import os
from segment import segment
import numpy as np
from PIL import Image
import cv2
from collections import Counter, OrderedDict
from data_loaders import BGRtoRGB
from sklearn.metrics import classification_report, accuracy_score
from baseline_KNN import run_baseline
import json

#Set of labels to retain from segmentation
keepers= ['person','chair','potted plant']
KNOWN = ['chairs', 'bottles', 'papers', 'books', 'desks', 'boxes', 'windows', 'exit-signs', 'coat-racks', 'radiators']
NOVEL = ['fire-extinguishers', 'desktop-pcs', 'electric-heaters', 'lamps', 'power-cables', 'monitors', 'people', 'plants', 'bins', 'doors' ]

n_support = 10
baseline = False  #Run KNN baseline or not?


def compute_similarity(qembedding, train_embeds):
    # Similarity matching against indexed data

    similarities = {}

    for emb_id, emb in train_embeds.items():
        # Cos sim reduces to dot product since embeddings are L2 normalized
        similarities[emb_id] = torch.mm(qembedding, emb.t()).item()

    # Return top-K results
    ranking = sorted(similarities.items(), key=lambda kv: kv[1], reverse=True)

    return ranking

def load_jsondata(path_to_json):

    """
    Assumes that the JSON is formatted as the output returned
    by the VGG Image Annotator (VIA)
    [http://www.robots.ox.ac.uk/~vgg/software/via/]

    :param path_to_json:
    :return: default path to all images, image data with regions and labels
    """

    with open(path_to_json, 'r') as jf:

        test_data = json.load(jf)

    return test_data["_via_settings"]["core"]["default_filepath"], OrderedDict(test_data['_via_img_metadata'])

def extract_base_embeddings(img_path, transforms, device, path_to_output):


    emb_space = {}


    for root, dirs, files in os.walk(img_path):

        if files:

            classname = str(root.split('/')[-1])


            for file in files:

                filename = str(file.split('/')[-1])

                emb_space[classname+'_'+filename] = base_embedding(os.path.join(root, file), device, transforms)


    with open(path_to_output,mode='wb') as outj:

        torch.save(obj=emb_space, f=outj)

    return emb_space

def KNN(input_e, all_embs, K, logfile):

    ranking = compute_similarity(input_e, all_embs)

    if K == 1:
        keyr, val = ranking[0]
        label = keyr.split("_")[0]

        print("The top most similar object is %s \n" % label)
        logfile.write("The top most similar object is %s \n" % label)

        print("With unique ID %s \n" % keyr)
        logfile.write("With unique ID %s \n" % keyr)

        return label

    else:
        print("The %i most similar objects to the provided image are: \n")
        logfile.write("The %i most similar objects to the provided image are: \n" % K)

        votes = Counter()
        ids = {}

        # Majority voting with discounts by distance from top position
        for k, (key, val) in enumerate(ranking[:K]):
            label = key.split("_")[0]  # [:-1]

            votes[label] += 1 / (k + 1)

            ids[label] = key

            print(label + ": " + str(val) + "\n")
            logfile.write(label + ": " + str(val) + "\n")
            print("With unique ID %s \n" % key)
            logfile.write("With unique ID %s \n" % key)

        win_label, win_score = max(votes.items(), key=lambda x: x[1])
        win_id = ids[win_label]

        if win_score > 1.0:

            print("The most similar object by majority voting is %s \n" % win_label)
            logfile.write("The most similar object by majority voting is %s \n" % win_label)
            print("With unique ID %s \n" % win_id)
            logfile.write("With unique ID %s \n" % win_id)

        else:

            print("Not sure about how to classify this object")
            logfile.write("Not sure about how to classify this object")

        return win_label


def test(model, model_checkpoint, data_type, path_to_test, path_to_bags, device, trans, path_to_train_embeds, K, N, sthresh= 1.0):



    #baseline KNN?
    if baseline:

        run_baseline(device, trans, K, N, n_support, KNOWN, path_to_train_embeds)

        return None

    # Code for test/inference time on one(few) shot(s)
    # for each query image

    if data_type == 'json':

        path_to_space = os.path.join(path_to_bags.split('test')[0], 'KMi_ref_embeds.dat')

        tgt_path = os.path.join(path_to_bags.split('test')[0], 'train')

        out_imgs = os.path.join(path_to_bags.split('test')[0], 'output_predictions')

        if not os.path.isfile(path_to_space):

            print("Creating embedding space")

            #Extract embeddings for reference images also taken in natural environment
            embedding_space = extract_base_embeddings(tgt_path, trans, device, path_to_space)

        else:

            print("Loading cached embedding space")
            embedding_space = torch.load(path_to_space) #, map_location={'cuda:0': 'cpu'})

        #For drawing the predictions
        all_classes = list(set([key.split('_')[0] for key in embedding_space.keys()]))
        COLORS = np.random.uniform(0, 255, size=(len(all_classes), 3))

        y_true = []
        y_pred = []

        cardinalities = Counter()

        base_path, img_collection = load_jsondata(path_to_bags)

        with open('pt_results/ranking_log.txt', 'a') as outr:

            for data_point in img_collection.values():

                img = cv2.imread(os.path.join(base_path, data_point["filename"]))

                if ".bag" in data_point["filename"]:
                    img = BGRtoRGB(img)

                # create copy to draw predictions on
                out_img = img.copy()

                print("%-----------------------------------------------------------------------% \n")
                print("Analyzing frame %s" % data_point["filename"])
                outr.write("Analyzing frame %s" % data_point["filename"])

                bboxes = data_point["regions"]

                #For each bounding box
                for region in bboxes:

                    #create a copy, crop it to region and store ground truth label

                    obj = img.copy()

                    box_data = region["shape_attributes"]
                    x = box_data["x"]
                    y = box_data["y"]
                    w = box_data["width"]
                    h = box_data["height"]

                    obj = obj[y:y+h,x:x+w]

                    cv2.imwrite('./temp.png', obj)
                    input_emb = base_embedding('./temp.png', device, trans)

                    gt_label = region["region_attributes"]["class"]

                    """
                    Visualize current object
                    
                    cv2.imshow('union', obj)
                    cv2.waitKey(2000)
                    cv2.destroyAllWindows()
                    """

                    cardinalities[gt_label] +=1
                    y_true.append(gt_label)

                    # Find (K)NN from input embedding
                    prediction= KNN(input_emb, embedding_space, K, outr)
                    y_pred.append(prediction)

                    #draw prediction
                    color = COLORS[all_classes.index(prediction)]
                    cv2.rectangle(out_img, (x, y), (x+w, y+h), color, 2)
                    if y-10 >0:
                        cv2.putText(out_img, prediction, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    else:
                        cv2.putText(out_img, prediction, (x - 10, y +h + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                print("%EOF---------------------------------------------------------------------% \n")
                cv2.imwrite(os.path.join(out_imgs, data_point["filename"]), out_img)
        #Check no. of instances per class
        #print(cardinalities)

        #Evaluation
        print("Class-wise test results \n")
        print(classification_report(y_true, y_pred))  # , target_names=all_classes))
        print(accuracy_score(y_true, y_pred))

        return None



    if data_type == 'pickled':

        train_embeds = torch.load(path_to_train_embeds, map_location={'cuda:0': 'cpu'})

        try:

            img_mat = np.load(path_to_bags, encoding = 'latin1', allow_pickle=True)
            #print(len(img_mat))
            #print(img_mat[0])
        except Exception as e:

            print("Problem while reading provided input +\n")
            print(str(e))
            return

        for img, timestamp in img_mat:

            """
            Convert from BRG -- OpenCV
            to RGB
            """
            #img2 = img.astype('float32')
            img = BGRtoRGB(img)

            #Write temporary img file
            cv2.imwrite('temp.jpg', img)

            with open('pt_results/ranking_log.txt', 'a') as outr:

                print("%-----------------------------------------------------------------------% \n")
                print("Analyzing frame %s" % timestamp)
                outr.write("Analyzing frame %s" % timestamp)

                predicted_boxes = segment('temp.jpg', img)


                for idx, (coords, segm_label) in enumerate(predicted_boxes):

                    print("Looking at %i -th object in this frame \n" % idx)


                    img_ = img.copy()
                    obj = img_[coords[1]:coords[3],coords[0]:coords[2]]

                    if str(obj) ==  "[]":

                        print("Empty bbox returned")
                        continue
                    #For each box wrapping an object

                    try:

                        #ob2 = BGRtoRGB(obj)
                        cv2.imshow('union', obj)
                        cv2.waitKey(1000)
                        cv2.destroyAllWindows()


                    except:

                        print(type(obj))
                        print(segm_label)
                    #Pre-process as usual validation/test images to extract embedding

                    qembedding = array_embedding(model, model_checkpoint, obj, \
                                                 device, transforms=trans)
                    """
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
                        #img_id = label +'_'+str(idx)+ '_'+ str(timestamp)
                        #train_embeds[img_id] = qembedding

                        #print("Keeping label produced by YOLO as %s \n" % yolo_label)
                    """

                    if segm_label in keepers:

                        #Just keeping prediction without adding to other embeddings
                        print("Spotted a %s \n" % segm_label)

                    else:

                        print("YOLO says this is a %s \n" % segm_label)

                    #Go ahead and classify by similarity

                    win_label = KNN(qembedding, train_embeds, K, outr)


                print("%EOF---------------------------------------------------------------------% \n")

        #Save updated embeddings after YOLO segmentation
        #with open(path_to_train_embeds, mode='wb') as outf:
        #    torch.save(obj=train_embeds, f=outf)

        #print("Updated embeddings saved under %s" % path_to_train_embeds)

        return None

    train_embeds = torch.load(path_to_train_embeds)

    with open('pt_results/ranking_log.txt', 'w') as outr:

        class_wise_res = []
        y_true = []
        y_pred = []
        kn_true = []
        kn_pred = []
        nw_true = []
        nw_pred = []

        #all_classes=[]


        if n_support == 10:

            path_to_support = path_to_test+"/../train"

        elif n_support == 5:

            path_to_support = path_to_test + "/../val"

        for root, dirs, files in os.walk(path_to_support):

            if files:

                classname = str(root.split('/')[-1])

                if classname in KNOWN:

                    continue

                for file in files:

                    filename = str(file.split('/')[-1])

                    #Update embedding space by mapping support set set as well
                    train_embeds[classname+'_'+filename] = path_embedding(model, model_checkpoint, os.path.join(root, file), \
                                        device, transforms=trans)

        #Then make predictions for all test examples (Known + Novel)
        for root, dirs, files in os.walk(path_to_test):

            if files:  #For each object class

                #tot_wrong = 0

                classname = str(root.split('/')[-1])

                outr.write(classname)
                #all_classes.append(classname)

                for file in files: #For each example in that class

                    y_true.append(classname)

                    if classname in KNOWN:

                        kn_true.append(classname)

                    else:

                        nw_true.append(classname)

                    print("%-----------------------------------------------------------------------% \n")
                    print("Looking at file %s \n" % file)
                    outr.write("Looking at file %s \n" % file)

                    qembedding = path_embedding(model, model_checkpoint, os.path.join(root, file), \
                                                 device, transforms=trans)

                    ranking = compute_similarity(qembedding, train_embeds)

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
                            #tot_wrong += 1

                        #avg_acc = correct_preds/K
                        class_accs.append(correct_preds)
                        label= win_label

                    y_pred.append(label)

                    if classname in KNOWN:

                        kn_pred.append(label)

                    else:

                        nw_pred.append(label)

                    print("%EOF---------------------------------------------------------------------% \n")

        #macro_avg = sum(class_accs)/len(class_accs)
        print("Class-wise test results \n")
        print(classification_report(y_true, y_pred )) #, target_names=all_classes))
        print(accuracy_score(y_true, y_pred))

        print("Known objects test results \n")
        print(classification_report(kn_true, kn_pred))  # , target_names=all_classes))
        print(accuracy_score(kn_true, kn_pred))

        print("Novel objects test results \n")
        print(classification_report(nw_true, nw_pred))  # , target_names=all_classes))
        print(accuracy_score(nw_true, nw_pred))

        #print('Mean average accuracy for class {} is {}'.format(classname, float(macro_avg)))
        #outr.write('Mean average accuracy for class {} is {}'.format(classname, float(macro_avg)))
        #class_wise_res.append((classname, macro_avg))

        return None #zip(*class_wise_res)