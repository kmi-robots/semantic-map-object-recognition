import torch
from embedding_extractor import path_embedding, array_embedding, base_embedding
import os
from segment import segment
import numpy as np
from PIL import Image
import cv2
import json
from collections import Counter, OrderedDict
import requests
import pandas
import time
from nltk.corpus import wordnet
from pattern.en import singularize, pluralize

from data_loaders import BGRtoRGB
from sklearn.metrics import classification_report, accuracy_score
from baseline_KNN import run_baseline


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

        print("Score: %f" % val)

        #For a proxy of confidence, we look at the top 5 in that ranking
        confs = Counter()

        for key, v in ranking[:5]:

            l = key.split("_")[0]
            #We sum the similarity scores obtained for that label
            confs[l] += val


        return label,confs[label], confs

    else:
        #print("The %i most similar objects to the provided image are: \n")
        #logfile.write("The %i most similar objects to the provided image are: \n" % K)

        votes = Counter()
        ids = {}

        # Majority voting with discounts by distance from top position
        for k, (key, val) in enumerate(ranking[:K]):

            label = key.split("_")[0]  # [:-1]

            votes[label] += val/(k + 1)    #nomin used to be 1

            ids[label] = key

            #print(label + ": " + str(val) + "\n")
            #logfile.write(label + ": " + str(val) + "\n")
            #print("With unique ID %s \n" % key)
            #logfile.write("With unique ID %s \n" % key)

        win_label, win_score = max(votes.items(), key=lambda x: x[1])

        """
        win_id = ids[win_label]
        if win_score > 1.0:

            print("The most similar object by majority voting is %s \n" % win_label)
            logfile.write("The most similar object by majority voting is %s \n" % win_label)
            print("With unique ID %s \n" % win_id)
            logfile.write("With unique ID %s \n" % win_id)

        else:

            print("Not sure about how to classify this object")
            logfile.write("Not sure about how to classify this object")
            
        """

        return win_label, win_score, votes

def compute_sem_sim(wemb1, wemb2):

    """
    Returns semantic similarity between two word embeddings
    based on Numberbatch 17.06 mini
    """

    dotprod = (wemb1*wemb2).sum()
    sq_wemb1 = wemb1.pow(2).sum()
    sq_wemb2 = wemb2.pow(2).sum()

    return (pow(sq_wemb1, 0.5)* pow(sq_wemb1, 0.5))/dotprod

def map(label):

    """
    Wordnet friendly format
    for specific objects
    TO-DO: make it less ugly/hardcoded
    """
    if label == "power-cables":

        label = "power-cords"

    elif label == "desktop-pcs":

        label = "computers"

    elif label == 'monitors':

        label = 'computer_monitors'

    elif label == "mugs":

        label = "cups"


    return label


def reverse_map(label):

    """
    Reverses the above for consistency on eval
    """

    if label == "power_cords":

        label= "power-cables"

    elif label == "computers":

        label= "desktop-pcs"

    elif label == "computer_monitors":

        label= "monitors"

    elif label == "cups":

        label = 'mugs'

    return label

def formatlabel(label):

    label = map(label)

    # make singular
    label = singularize(label)

    if '-' in label:

        label = label.replace('-', '_')

        wn_label = label.replace('-', '')
    else:

        wn_label = label


    return label, get_synset(wn_label)   #[:-1]  # remove plurals


def get_synset(class_label):

    """
    Returns the wordnet synset for a given term, if any is found
    """

    syns = wordnet.synsets(class_label)

    if syns:

        return syns[0]
    else:

        return None

def query_conceptnet(class_set, path_to_dict):

    """
    This dictionary of all possible score combinations is
    pre-computed only on the first time running the script
    """

    #numberbatch_vecs = pandas.read_hdf(path_to_dict.split("KMi_conceptrel.json")[0]+'mini.h5')
    base = '/c/en/'

    score_dict = {}

    for term in class_set:

        print(term)

        #all other classes but itself

        term_dict ={}

        #emb1 = numberbatch_vecs.loc[base + term]

        for term2 in [c for c in class_set if c != term]: #Skip identity case

            try:

                node = requests.get('http://api.conceptnet.io/relatedness?node1='+\
                                base+term+'&node2='+base+term2).json()


            except:

                time.sleep(60)

                try:
                    node = requests.get('http://api.conceptnet.io/relatedness?node1=' + \
                                    base + term + '&node2=' + base + term2).json()

                except:

                    print(requests.get('http://api.conceptnet.io/relatedness?node1=' + \
                                        base + term + '&node2=' + base + term2))


            print(node['value'])
            term_dict[term2] = node['value']

            """
            #Using the local embeds per se does not handle OOV
            emb2 = numberbatch_vecs.loc[base + term2]

            term_dict[term2] = compute_sem_sim(emb1,emb2)
            """

        #Relatedness scores for that term, in ascending order
        term_dict = sorted(term_dict.items(), key=lambda kv: kv[1])#, reverse=True)
        score_dict[term] = OrderedDict(term_dict)

    with open(path_to_dict, 'w') as jf:

        json.dump(score_dict, jf)

    return score_dict


def call_cp_api(word1, word2):

    try:

        outcome = requests.get('http://api.conceptnet.io/relatedness?node1=/c/en/' + \
                     word1 + '&node2=/c/en/' + word2).json()

        return outcome['value']

    except:

        time.sleep(60)

        outcome = requests.get('http://api.conceptnet.io/relatedness?node1=/c/en/' + \
                               word1 + '&node2=/c/en/' + word2).json()

        return outcome['value']



def conceptnet_relatedness(subject, candidates, object):

    base_score = call_cp_api(subject, object)

    pred_subject = subject

    #print(base_score)
    #Is there any other label in the ranking making more sense?

    for o_class, confidence in candidates.items():

        f_class,_ = formatlabel(o_class)


        if f_class == subject:

            continue #Skip the object itself

        score = call_cp_api(f_class, object)

        if score > base_score:

            base_score = score
            pred_subject = o_class

    print("CONCEPTNET: Within the ranking, the most likely subject is %s" % pred_subject)
    if singularize(pred_subject) == pred_subject:

        #Re-format back for evaluation
        pred_subject = pluralize(pred_subject)

    pred_subject = reverse_map(pred_subject)

    return pred_subject.replace('_', '-'), base_score


def extract_spatial(weak_idx, obj_list, VG_base=None):

    preds,confs, coords, rankings = zip(*obj_list)

    weak_pred, weak_synset = formatlabel(preds[weak_idx])

    #weak_conf = confs[weak_idx]
    x_a= coords[weak_idx][0]
    y_a = coords[weak_idx][1]
    w_a = coords[weak_idx][2]
    h_a = coords[weak_idx][3]
    weak_rank = rankings[weak_idx]

    new_ranking = Counter()

    #rel_graph["subject"] = weak_pred
    #rel_graph["synset"] = weak_synset

    #Are any other objects are nearby?
    for label, score, bbox, ranking in [obj_list[i] for i in range(len(obj_list)) if i!= weak_idx]:

        #Add threshold: only link with object we are reasonably confident about
        if score >= 1.14: #2.50: more than 50% confident about the other object

            x_b = bbox[0]
            y_b = bbox[1]
            w_b = bbox[2]
            h_b = bbox[3]

            #Spatial ref is relative to anchor object
            spatial_rel = check_spatial(x_a, x_b, y_a, y_b, w_a, h_a, w_b, h_b)

            if spatial_rel is not None:

                sem_obj, obj_syn = formatlabel(label)

                #Verify on supporting knowledge if found relation makes sense

                if weak_synset and obj_syn:

                    print(weak_pred+"( "+weak_synset.name()+" ) "+spatial_rel+" "+sem_obj+"( "+obj_syn.name()+" ) \n")

                else:

                    print(weak_pred+" "+spatial_rel+" "+sem_obj+"\n")


                #Does this relation make sense?
                #1) In conceptnet?
                cp_pred, cp_score = conceptnet_relatedness(weak_pred, weak_rank, sem_obj)

                #Changed: no first filter by max


                # print(base_score)
                # Is there any other label in the ranking making more sense?

                """
                for o_class, confidence in weak_rank.items():

                    f_class, _ = formatlabel(o_class)

                    \"""
                    if f_class == weak_pred:
                        
                        continue  # Skip the object itself

                    \"""

                    cp_score = call_cp_api(f_class, sem_obj)

                    \"""  
                       if score > base_score:
                           base_score = score
                           pred_subject = o_class
    
                    \"""

                    if singularize(o_class) == o_class:
                        # Re-format back for evaluation
                        o_class = pluralize(o_class)

                    o_class = reverse_map(o_class)

                    o_class.replace('_', '-')

                    new_ranking[o_class] += cp_score
                """
                new_ranking[cp_pred] += cp_score

    return new_ranking

FLOOR = get_synset("floor").name()
TABLE = wordnet.synsets("table")[1].name() #table.n.01 is a data table
                                            # While the piece of furniture is table.n.02


def check_horizontal(y, h, img_res= (1280,720)):

    bar_y = y + h #/2  #Replace bar_y with actual lower y
    thresh = img_res[1] - img_res[1]/3

    if bar_y > thresh and bar_y <= img_res[1]:

        return "on "+ FLOOR

    else:

        return None


def check_spatial(x1, x2, y1, y2, w1, h1, w2, h2):

    #CHANGED: make horizontal bbox expansion relative to each bbox size as opposed to absolute px value
    lowto = 10
    hito = w1

    bar_x2 = x2 + w2/2
    bar_y2 = y2 + h2/2


    if bar_y2 > y1 + h1 and bar_x2 >= x1 - hito and bar_x2 <= x1 + w1 + hito:

        return "on"

    elif bar_y2 < y1 and bar_x2 >= x1 - hito and bar_x2 <= x1 + w1 + hito:

        return "under"

    elif bar_x2 <= x1+w1+hito and bar_x2 >= x1 - hito and bar_y2 <= y1+h1+lowto and bar_y2>= y1 - lowto:

        return "near"

    else:

        return None



def replace(ranking, coords, VG_onrel, frame_objs, weak_idx, replaced = False):

    for l, s in ranking.most_common():

        qlab, syn = formatlabel(l)

        # Which is the top score alternative object not usually on the floor?
        if syn:
            try:

                floor = VG_onrel[str((syn.name(), FLOOR))]

                try:
                    table = VG_onrel[str((syn.name(), TABLE))]

                    if table >= floor:
                        # Replace
                        frame_objs[weak_idx] = (l, s, coords, ranking)

                        print("Replaced with " + qlab + " instead")

                        replaced = True

                        return frame_objs, replaced

                except:

                    continue


            except KeyError:

                # Replace
                frame_objs[weak_idx] = (l, s, coords, ranking)

                print("Replaced with " + qlab + " instead")

                replaced = True

                return frame_objs, replaced


    return frame_objs, replaced

def proxy_floor(object, VG_base):

    objl, objsyn = formatlabel(object)
    VG_onrel = VG_base["relations"]

    if objsyn:

        try:

            on_floor = VG_onrel[str((objsyn.name(), FLOOR))]

            try:

                on_table = VG_onrel[str((objsyn.name(), TABLE))]

                if (on_floor > on_table +80):

                    return "on "+ FLOOR

                else:

                    return "both"

            except KeyError:

                return "on "+ FLOOR

        except KeyError:

            return None

    else:

        return 'no synset'

def correct_floating(o_y, o_h, weak_idx, frame_objs, VG_base, rflag=False):

    floor_rel = check_horizontal(o_y, o_h)

    label, score, coords, ranking = frame_objs[weak_idx]

    qlabel, synset = formatlabel(label)

    VG_onrel = VG_base["relations"]

    if synset:

        #not applicable to Health and Safety objects or other OOV

        try:

            on_floor = VG_onrel[str((synset.name(), FLOOR))]

            try:

                on_table = VG_onrel[str((synset.name(), TABLE))]

                if (on_floor > on_table) and not floor_rel:

                    frame_objs, rflag = replace(ranking, coords, VG_onrel, frame_objs, weak_idx)

                elif floor_rel:

                    print(qlabel + " makes sense on the floor")

                else:

                    print(qlabel + " makes sense also above the floor ")

            except KeyError:

                if not floor_rel:

                    print(qlabel + " could be floating when it shouldn't ")
                    # needs to be corrected: potentially floating object

                    frame_objs, rflag = replace(ranking, coords, VG_onrel, frame_objs, weak_idx)


                else:

                    print(qlabel + " makes sense on the floor ")


        except KeyError:

            #Does not have on floor rels in VG
            print("Not doing anything ")
            pass


    return frame_objs, rflag




def show_leastconf(scene_objs):


    preds,confs, coords, rankings = zip(*scene_objs)

    if min(confs) < 1.14: #2.5:  #less than 50%

        i = confs.index(min(confs))

        print("Weakest prediction was \n")
        print(preds[i])
        print("With associated ranking scores \n")
        print(rankings[i])

        return i

    else:

        return None




def test(model, model_checkpoint, data_type, path_to_test, path_to_bags, device, trans, path_to_train_embeds, K, N):


    #baseline KNN?
    if baseline:

        run_baseline(device, trans, K, N, n_support, KNOWN, path_to_train_embeds)

        return None

    # Code for test/inference time on one(few) shot(s)
    # for each query image

    if data_type == 'json':

        path_to_space = os.path.join(path_to_bags.split('KMi_collection')[0], 'kmish25/embeddings_imprKNET_1prod.dat') #embeddings_imprKNET_1prod_DA_static #os.path.join(path_to_bags.split('test')[0], 'KMi_ref_embeds.dat')
        path_to_state = os.path.join(path_to_bags.split('KMi_collection')[0], 'kmish25/checkpoint_imprKNET_1prod.pt') #checkpoint_imprKNET_1prod_DA_static

        #path_to_basespace = os.path.join(path_to_bags.split('test')[0], 'KMi_ref_embeds.dat')

        tgt_path = os.path.join(path_to_bags.split('test')[0], 'train')
        out_imgs = os.path.join(path_to_bags.split('test')[0], 'output_predictions')

        path_to_concepts = os.path.join(path_to_bags.split('KMi_collection')[0],'numberbatch/KMi_conceptrel.json')
        path_to_VG = os.path.join(path_to_bags.split('KMi_collection')[0],'visual_genome/filtered_spatial.json')


        if not os.path.isdir(out_imgs):

            os.mkdir(out_imgs)

        if os.path.isfile(path_to_VG):

            start = time.time()
            print("Loading Visual Genome spatial relations...")
            with open(path_to_VG, 'r') as jin:
                VG_data = json.load(jin)

            print("Took %f seconds " % (time.time()-start))

        """
        if not os.path.isfile(path_to_basespace):

            print("Creating embedding space")

            #Extract embeddings for reference images also taken in natural environment
            base_embedding_space = extract_base_embeddings(tgt_path, trans, device, path_to_space)

        else:

            print("Loading cached embedding space")
            base_embedding_space = torch.load(path_to_space, map_location={'cuda:0': 'cpu'})
        """
        embedding_space = torch.load(path_to_space, map_location={'cuda:0': 'cpu'})


        #For drawing the predictions
        all_classes = list(set([key.split('_')[0] for key in embedding_space.keys()]))

        all_classes = sorted(all_classes)

        COLORS = np.random.uniform(0, 255, size=(len(all_classes), 3))

        """
        if not os.path.isfile(path_to_concepts):

            print("Creating dictionary of relatedness between classes...")
            relat_dict = query_conceptnet(qall_classes, path_to_concepts)

        else:

            print("Loading cached dictionary of relatedness between classes...")
            with open(path_to_concepts, 'r') as jf:
                relat_dict = json.load(jf)

        """
        y_true = []
        y_pred = []

        cardinalities = Counter()

        base_path, img_collection = load_jsondata(path_to_bags)

        data = list(reversed(img_collection.values()))

        with open('pt_results/ranking_log.txt', 'a') as outr:

            for data_point in data: #reversed(img_collection.values()):

                img = cv2.imread(os.path.join(base_path, data_point["filename"]))

                if ".bag" in data_point["filename"]:
                    img = BGRtoRGB(img)

                # create copy to draw predictions on
                out_img = img.copy()
                out_VG = img.copy()
                out_CP = img.copy()

                print("%-----------------------------------------------------------------------% \n")
                print("Analyzing frame %s" % data_point["filename"])
                outr.write("Analyzing frame %s" % data_point["filename"])

                bboxes = data_point["regions"]
                frame_objs = []

                if not bboxes:

                    print("NOT annotated")

                    continue

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
                    input_emb = array_embedding(model, path_to_state, obj, device, transforms=trans)


                    """
                    cv2.imwrite('./temp.png', obj)
                    basein_emb = base_embedding('./temp.png', device, trans)
                    """

                    gt_label = region["region_attributes"]["class"]


                    #Visualize current object
                    """
                    cv2.imshow('union', obj)
                    cv2.waitKey(5000)
                    cv2.destroyAllWindows()
                    """

                    cardinalities[gt_label] +=1
                    y_true.append(gt_label)

                    # Find (K)NN from input embedding
                    """
                    print("%%%%%%%The baseline NN predicted %%%%%%%%%%%%%%%%%%%%%")
                    baseline_pred = KNN(basein_emb, base_embedding_space, K, outr)
                    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
                    """

                    #print("%%%%%%%The trained model predicted %%%%%%%%%%%%%%%%%%%%%")
                    prediction, conf, rank_confs = KNN(input_emb, embedding_space, K, outr)
                    #print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%")


                    frame_objs.append((prediction, conf, (x,y,w,h), rank_confs))
                    #y_pred.append(prediction)

                    #draw prediction
                    color = COLORS[all_classes.index(prediction)]
                    cv2.rectangle(out_img, (x, y), (x+w, y+h), color, 2)


                    if y-10 >0:
                        cv2.putText(out_img, prediction+"  "+str(round(conf,2)), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    else:
                        cv2.putText(out_img, prediction+"  "+str(round(conf,2)), (x - 10, y +h + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                print("%EOF---------------------------------------------------------------------% \n")

                """
                cv2.imshow('union', out_img)
                cv2.waitKey(5000)
                cv2.destroyAllWindows()

                """

                """Correcting least confident predictions by querying external knowledge"""


                weak_idx = show_leastconf(frame_objs)


                if weak_idx is not None:

                    orig_label, _, coords, _ = frame_objs[weak_idx]


                    if VG_data:

                        #Pre-correction based on a subset of relations in Visual Genome
                        o_y = coords[1]
                        o_h = coords[3]

                        frame_objs, corrected_flag = correct_floating(o_y, o_h, weak_idx, frame_objs, VG_data['on'])

                        if corrected_flag:

                            #Skip further reasoning
                            corr_preds, _, _, _ = zip(*frame_objs)
                            y_pred.extend(corr_preds)
                            """
                            #And show corrected image 
                            for lb,cf,(x,y,w,h), rank in frame_objs:

                                color = COLORS[all_classes.index(lb)]
                                cv2.rectangle(out_VG, (x, y), (x + w, y + h), color, 2)

                                if y - 10 > 0:
                                    cv2.putText(out_VG, lb, (x - 10, y - 10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                else:
                                    cv2.putText(out_VG, lb, (x - 10, y + h + 10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                            cv2.imwrite(os.path.join(out_imgs, 'Kground_ImprintedKNET', data_point["filename"]), out_VG)
                            """
                            continue


                    if len(frame_objs) <= 1:

                        print("Only one object found in this scene...skipping contextual reasoning")
                        corr_preds, _, _, _ = zip(*frame_objs)
                        y_pred.extend(corr_preds)

                        #cv2.imwrite(os.path.join(out_imgs, 'ImprintedKNET', data_point["filename"]), out_img)

                        continue

                    new_preds = extract_spatial(weak_idx, frame_objs, VG_base=VG_data)

                    if new_preds:

                        #Take the max w.r.t. semantic relatedness
                        wlabel, wscore = max(new_preds.items(), key=lambda x: x[1])

                        wlabel = reverse_map(wlabel)

                        print("Based on all nearby objects this is a %s" %wlabel)
                        print("With confidence %f" % wscore)

                        print(sorted(new_preds.items(), key=lambda x: x[1], reverse=True))

                        # is the rel VG-validated w.r.t. floor?
                        orig_floor = proxy_floor(orig_label, VG_data['on'])
                        corrected_floor_out =proxy_floor(wlabel, VG_data['on'])

                        
                        #ONLY if so (or if originally OOV) change it
                        if orig_floor=='no synset' or orig_floor == corrected_floor_out:

                            print("Accepting proposed correction")
                            frame_objs[weak_idx]= (wlabel.replace('_','-'), wscore, coords, new_preds)
                        else:

                            print("Rejecting suggested correction: replacement does not make sense w.r.t floor")


                        """
                        for lb, cf, (x, y, w, h), rank in frame_objs:

                            color = COLORS[all_classes.index(lb)]
                            cv2.rectangle(out_VG, (x, y), (x + w, y + h), color, 2)

                            if y - 10 > 0:
                                cv2.putText(out_VG, lb, (x - 10, y - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            else:
                                cv2.putText(out_VG, lb, (x - 10, y + h + 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                        cv2.imwrite(os.path.join(out_imgs, 'Kground_ImprintedKNET', data_point["filename"]), out_VG)
                        """
                corr_preds, _ , _, _= zip(*frame_objs)
                y_pred.extend(corr_preds)

                #print("Off you go")
                # And show corrected image
                """
                for lb, cf, (x, y, w, h), rank in frame_objs:

                    color = COLORS[all_classes.index(lb)]
                    cv2.rectangle(out_CP, (x, y), (x + w, y + h), color, 2)

                    if y - 10 > 0:
                        cv2.putText(out_CP, lb, (x - 10, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    else:
                        cv2.putText(out_CP, lb, (x - 10, y + h + 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    

                cv2.imwrite(os.path.join(out_imgs, 'ImprintedKNET_ConceptRel', data_point["filename"]), out_CP)
                """

                #cv2.imwrite(os.path.join(out_imgs, 'ImprintedKNET', data_point["filename"]), out_img)

        #Check no. of instances per class
        #print(cardinalities)
        print(len(y_pred)==len(y_true))

        #Evaluation
        print("Class-wise test results \n")
        print(classification_report(y_true, y_pred))  # , target_names=all_classes))
        print(accuracy_score(y_true, y_pred))

        #print(COLORS)
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