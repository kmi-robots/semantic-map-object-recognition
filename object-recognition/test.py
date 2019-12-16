import torch
from embedding_extractor import path_embedding, array_embedding, base_embedding
import os
from segment import segment, display_mask
import numpy as np
from PIL import Image
import cv2
import json
from collections import Counter, OrderedDict
import time


from common_sense import show_leastconf, correct_floating, extract_spatial, proxy_floor, reverse_map
from data_loaders import BGRtoRGB
from sklearn.metrics import classification_report, accuracy_score
from baseline_KNN import run_baseline
from segment import white_balance
from spatial import find_real_xyz


#Set of labels to retain from segmentation
keepers= ['person','chair','potted plant']

KNOWN = ['chairs', 'bottles', 'papers', 'books', 'desks', 'boxes', 'windows', 'exit-signs', 'coat-racks', 'radiators']
NOVEL = ['fire-extinguishers', 'desktop-pcs', 'electric-heaters', 'lamps', 'power-cables', 'monitors', 'people', 'plants', 'bins', 'doors' ]

n_support = 10


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

def KNN(input_e, all_embs, K, voting):

    ranking = compute_similarity(input_e, all_embs)

    if K == 1:

        keyr, val = ranking[0]
        label = keyr.split("_")[0]

        print("The top most similar object is %s \n" % label)

        print("With unique ID %s \n" % keyr)

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

            if voting =='discounted':

                votes[label] += val/(k + 1)

            else:
                #Just picking the class that gets the majority of votes
                votes[label] += 1 #/ (k + 1)

            ids[label] = key

            #print(label + ": " + str(val) + "\n")
            #logfile.write(label + ": " + str(val) + "\n")
            #print("With unique ID %s \n" % key)
            #logfile.write("With unique ID %s \n" % key)

        #print(str(ranking[:K]))
        #print("RANKING Discounted majority voting")
        #print(str(votes))

        win_label, win_score = max(votes.items(), key=lambda x: x[1])

        """
        print("The winner is "+win_label)
        print("With confidence score " + win_score)
        
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



def test(data_type, path_to_input,  args, model, device, trans, camera_img = None, path_to_train_embeds=None, KBrules=None):

    K = args.K
    N = args.N

    baseline = True if args.stage == 'baseline' else False

    #baseline KNN?
    if baseline:

        run_baseline(device, trans, K, N, n_support, KNOWN, path_to_train_embeds)

        return None

    # Code for test/inference time on one(few) shot(s)
    # for each query image

    out_imgs = os.path.join(path_to_input.split('test')[0], 'output_predictions')


    if not os.path.isdir(out_imgs):

        os.mkdir(out_imgs)

        # For drawing the predictions

    y_true = []
    y_pred = []

    cardinalities = Counter()


    # sem = args.sem if args.sem != 'none' else None

    voting = args.Kvoting

    path_to_space = os.path.join(path_to_input.split('KMi_collection')[0],
                                 'kmish25/embeddings_imprKNET_1prod.dat')  # embeddings_imprKNET_1prod_DA_static #os.path.join(path_to_bags.split('test')[0], 'KMi_ref_embeds.dat')
    path_to_state = os.path.join(path_to_input.split('KMi_collection')[0],
                                 'kmish25/checkpoint_imprKNET_1prod.pt')  # checkpoint_imprKNET_1prod_DA_static

    # path_to_basespace = os.path.join(path_to_bags.split('test')[0], 'KMi_ref_embeds.dat')

    # tgt_path = os.path.join(path_to_input.split('test')[0], 'train')
    # path_to_concepts = os.path.join(path_to_input.split('KMi_collection')[0], 'numberbatch/KMi_conceptrel.json')
    """
     if not os.path.isfile(path_to_concepts):

         print("Creating dictionary of relatedness between classes...")
         relat_dict = query_conceptnet(qall_classes, path_to_concepts)

     else:

         print("Loading cached dictionary of relatedness between classes...")
         with open(path_to_concepts, 'r') as jf:
             relat_dict = json.load(jf)

    """
    path_to_VG = os.path.join(path_to_input.split('KMi_collection')[0], 'visual_genome/filtered_spatial.json')
    embedding_space = torch.load(path_to_space, map_location={'cuda:0': 'cpu'})

    all_classes = list(set([key.split('_')[0] for key in embedding_space.keys()]))

    all_classes = sorted(all_classes)

    COLORS = np.random.uniform(0, 255, size=(len(all_classes), 3))

    VG_data = None

    if args.sem!='none' and os.path.isfile(path_to_VG):

        start = time.time()
        print("Loading Visual Genome spatial relations...")
        with open(path_to_VG, 'r') as jin:
            VG_data = json.load(jin)

        print("Took %f seconds " % (time.time() - start))

    img_collection = OrderedDict()

    if data_type == 'camera':

        # Data acquired from camera online
        # Initialises and returns loaded data spaces only once

        return VG_data, embedding_space, cardinalities,COLORS, all_classes


    elif data_type == 'json':

        #Path points to a VIA Annotated JSON file
        base_path, img_collection = load_jsondata(path_to_input)
        #data = list(reversed(img_collection.values()))#[16:]


    elif data_type == 'pickled':

        # Path points to pickled file, e.g., after extracting and converting data from a rosbag - for offline processing
        # Can be used for the output of the utils/bag_converter.py script
        base_path = path_to_input

        try:

            img_mat = np.load(path_to_input, encoding='latin1', allow_pickle=True)

        except Exception as e:

            print("Problem while reading provided input +\n")
            print(str(e))
            return

        #create list of data records from img_mat
        for img, timestamp in img_mat:

            node = OrderedDict()
            node["filename"] = timestamp
            node["regions"] = None
            node["data"] = img

            img_collection.update(node)

    else:

        base_path = path_to_input
        # Images read from local file

        for root, dirs, files in os.walk(path_to_input):

            if files:

                for file in files:  # For each example in that class

                    node = OrderedDict()
                    node["filename"] = os.path.join(root,file)
                    node["regions"] = None

                    img_collection.update(node)

    norm_coeff = 0

    if voting == "discounted":

        for k in range(1, K+1):

            norm_coeff += 1/k

    else:

        norm_coeff = K


    #Image Processing------------------------------------------------------------------------------------------
    #if data_type !='camera':

    data = img_collection.values()
    #batch of multiple images to process
    for data_point in list(reversed(data)):  # list(reversed(data))[:15]:

        _, y_pred, y_true, run_eval= img_processing_pipeline(data_point, base_path, args, model, device, trans, cardinalities, COLORS, all_classes, \
                                                           K, args.sem, voting, VG_data, y_true, y_pred, embedding_space, KBrules=KBrules, norm_coeff=norm_coeff)


    #Evaluation------------------------------------------------------------------------------------------------

    #Only if annotated ground truth is available
    if run_eval:

        print("Class-wise test results \n")
        print(classification_report(y_true, y_pred))  # , target_names=all_classes))
        print(accuracy_score(y_true, y_pred))


    return None


def img_processing_pipeline(data_point, base_path, args, model, device, trans, cardinalities, COLORS, all_classes, \
                                                           K, sem, voting, VG_data, y_true, y_pred, embedding_space, extract_SR=True, VQA=False, \
                                                           KBrules =None, norm_coeff=None):
    #print(type(data_point))

    try:

        img = data_point["data"]

    except Exception as e:
        print(e)
        img = cv2.imread(os.path.join(base_path, data_point["filename"]))

    #cv2.imshow('union', img)
    #cv2.waitKey(10000)
    #cv2.destroyAllWindows()

    # if ".bag" in data_point["filename"]:
    if ".bag" in data_point["filename"]:
        img = BGRtoRGB(img)

    # create copy to draw predictions on

    #import matplotlib.pyplot as plt
    #plt.imshow(img)
    #plt.show()

    denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15)
    denoised = BGRtoRGB(white_balance(denoised))


    #plt.imshow(denoised)
    #plt.show()

    out_img = denoised.copy()
    out_VG = denoised.copy()
    #out_CP = img.copy()

    
    print("%-----------------------------------------------------------------------% \n")
    print("Analyzing frame %s" % data_point["filename"])

    whole_start = time.time()

    #Extract foreground from pcl
    try:
        pcl = data_point["pcl"]

    except KeyError:
        #method not accessed from camera input
        pcl=None

    frame_objs = []
    colour_seq = []
    locations = []
    hsv_colors = []
    add_properties = []

    if data_point["regions"] is not None:

        # rely on ground truth boxes
        bboxes = data_point["regions"]
        frame_objs = []
        predicted_masks = None
        run_eval = True

    else:

        # segment it
        
        predicted_boxes, predicted_masks = segment(denoised, w_saliency=False,  depth_image= data_point["depth_image"])

        try:

            bboxes, yolo_labels = zip(*predicted_boxes)

        except ValueError:
            #no bboxes returned
            bboxes= None
            yolo_labels = None

        run_eval = False

    if bboxes is not None:

        # For each bounding box
        for idx, region in enumerate(bboxes):

            # create a copy, crop it to region and store ground truth label

            obj = denoised.copy() #change img to denoised to see effect of denoising and white balancing on predictions
            segm_label = ''

            if args.bboxes == 'true':

                box_data = region["shape_attributes"]
                x = box_data["x"]
                y = box_data["y"]
                x2 = x + box_data["width"]
                y2 = y + box_data["height"]

            else:

                segm_label = yolo_labels[idx]

                x = region[0]
                x2 = region[2]
                y = region[1]
                y2 = region[3]

            box_area = (x2-x)*(y2-y)

            obj = obj[y:y2, x:x2]


            input_emb = array_embedding(model, obj, device, transforms=trans)

            if input_emb is None:

                # Image could not be processed/ embedding could not be generated for some reason, skip obj
                continue

                #return [out_img, out_VG, None], y_pred, y_true, run_eval, SR_KB

            """
            cv2.imwrite('./temp.png', obj)
            basein_emb = base_embedding('./temp.png', device, trans)
            """
            try:
                gt_label = region["region_attributes"]["class"]

            except:
                # not annotated yet
                gt_label = 'N/A'

            # Visualize current object
            """
            cv2.imshow('union', obj)
            cv2.waitKey(5000)
            cv2.destroyAllWindows()
            """

            cardinalities[gt_label] += 1
            y_true.append(gt_label)

            # Find (K)NN from input embedding
            """
            print("%%%%%%%The baseline NN predicted %%%%%%%%%%%%%%%%%%%%%")
            baseline_pred = KNN(basein_emb, base_embedding_space, K, outr, voting)
            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            """

            # print("%%%%%%%The trained model predicted %%%%%%%%%%%%%%%%%%%%%")
            prediction, conf, rank_confs = KNN(input_emb, embedding_space, K, voting)
            # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

            #Extract colour features too
            from segment import segment_by_color

            pix_counter = segment_by_color(obj)

            hsv_colors.append(pix_counter)

            frame_objs.append((prediction, conf, (x, y, x2, y2), rank_confs))


            #Pinpoint object in "real world"
            try:

                vertices_x, vertices_y, vertices_z = find_real_xyz(x,y,x2,y2, pcl, img.shape, )
                locations.append((vertices_x, vertices_y, vertices_z))
                vertices_coords = list(zip(vertices_x, vertices_y, vertices_z))
                has_loc= True

            except AssertionError:

                #Data not coming from robot, location not available
                locations.append([None,None,None])
                has_loc = False

            # draw prediction
            color = COLORS[all_classes.index(prediction)]
            colour_seq.append(color)
            cv2.rectangle(out_img, (x, y), (x2, y2), color, 2)

            if y - 10 > 0:

                cv2.putText(out_img, prediction + "  " + segm_label + "  "+str(round(conf, 2)), (x - 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            else:

                cv2.putText(out_img, prediction + "  " + segm_label + "  "+str(round(conf, 2)), (x - 10, y2 + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # draw semantic masks if any (supports pytorch masks only for now)
            """
            if predicted_masks is not None:
                # print(predicted_masks[idx])
                # print(predicted_masks[idx].shape)
                mask = predicted_masks[idx]

                out_img = display_mask(out_img, mask, color)
            """
            # cv2.imshow('union', obj)
            # cv2.waitKey(10000)
            # cv2.destroyAllWindows()


            #### Extract other properties and discretize if in reason mode#####
            if args.stage == "reason":

                import spatial

                u = int(x + (x2 - x) / 2)
                v = int(y + (y2 - y) / 2)

                # discrete centroid position in 2d frame
                if (v <= spatial.TOP_TH):
                    pos = "top"

                elif v > spatial.TOP_TH <= spatial.MIDDLE_TH:

                    pos = "middle"

                elif v > spatial.BTM_TH:

                    pos = "bottom"

                if has_loc:

                    IMG_AREA = spatial.IMG_AREA
                    """discretize relative distance: camera range(0.60,8.0) metres"""

                    uv_z = vertices_coords[0][-1]

                    if uv_z <= 3.06:

                        dis = "close"

                    elif uv_z <= 5.52:

                        dis = "distant"

                    else:

                        dis = "far"

                else:
                    # This property will not be avialble for KMiSet 25 pre-annotated
                    # as it is based on depth data
                    IMG_AREA = 1280 * 720

                    dis = None


                if float(box_area/IMG_AREA) <= 0.33:
                    size = 'small'

                elif float(box_area/IMG_AREA) <= 0.66:
                    size = 'medium'

                else:
                    size = "large"

                dom_colour, colour_val = pix_counter.most_common()[0]

                add_properties.append([(u,v), pos, size, dom_colour, dis])


            vqastart = time.time()
            if VQA:
                # Also ask Pythia about this object
                from pythia_demo import PythiaDemo

                demo = PythiaDemo()

                question_text = "is this a "+prediction

                scores, predictions = demo.predict(obj, question_text)

                winner, w_score = (predictions[0], scores[0])

                if winner =="yes":

                    print("Pythia says that object IS a "+ prediction)

                else:

                    print("Pythia says that object IS NOT a "+ prediction)

                print("Took %f sec for Pythia to guess" % float(time.time() - vqastart))



        # Correction via internal KB ---------------------------------------------------------------------------

        if args.stage == "reason":

            # Find near objects in scene
            import spatial
            from spatial import extract_near_list

            for i, (prediction, conf, coords, rank_confs) in enumerate(frame_objs):

                (u,v), pos, size, dom_colour, dis = add_properties[i]

                near_obj_labels = extract_near_list((u,v), [box_prop for k, box_prop in enumerate(frame_objs) if k!=i])

                add_properties[i] = [(u,v), pos, size, dom_colour, dis, near_obj_labels]

                mod_rank = Counter()

                for class_name, cum_score in rank_confs.most_common():

                    norm_score = cum_score/norm_coeff

                    #Modify confidence ranking given the rules available in KBrules
                    subset = KBrules[KBrules["consequents"] == {class_name}]

                    if not subset.empty: # if any rule is found for that object (as consequent)

                        if subset.shape[0]> 1: # if more then one rule is found
                        #pick the best one (e.g., highest leverage)

                            mod_score = norm_score * subset.loc[subset["leverage"].idxmax()].confidence

                        else:

                            mod_score = norm_score * subset.confidence

                    else:

                        #keep normalised version of old (Vision-based) score otherwise
                        mod_score = norm_score

                    # it's a counter so it will be a unique set of classes, we can add value for key only once
                    mod_rank[class_name] = mod_score

                win_l, win_score = mod_rank.most_common()[0]

                #new predictions based on modified scores
                if win_l == prediction:

                    print("Prediction kept as %s" % win_l)
                else:
                    print("Corrected with %s instead based on KB rules" % win_l)
                y_pred.append(win_l)



        # Correction via ConceptNet + VG -----------------------------------------------------------------------
        """Correcting least confident predictions by querying external knowledge"""
        if sem !='none':# Only if semantic modules are to be included

            weak_idx = show_leastconf(frame_objs)

            if weak_idx is not None:

                orig_label, _, coords, _ = frame_objs[weak_idx]

                if VG_data and sem == 'full':

                    # Pre-correction based on a subset of relations in Visual Genome
                    o_y = coords[1]
                    o_h = coords[3]

                    frame_objs, corrected_flag = correct_floating(o_y, o_h, weak_idx, frame_objs, VG_data['on'])

                    if corrected_flag:
                        # Skip further reasoning
                        corr_preds, _, _, modified_rank = zip(*frame_objs)

                        y_pred.extend(corr_preds)

                        print("Ranking for weakest object changed into")
                        print(str(modified_rank))

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

                        """
                        cv2.imwrite(os.path.join(out_imgs, 'Kground_ImprintedKNET', data_point["filename"]), out_VG)
                        """
                        return [out_img, out_VG, frame_objs, colour_seq, locations, hsv_colors], y_pred, y_true, run_eval  # Continue to next image

                if len(frame_objs) <= 1:

                    print("Only one object found in this scene...skipping contextual reasoning")
                    corr_preds, _, _, _ = zip(*frame_objs)
                    y_pred.extend(corr_preds)

                    # cv2.imwrite(os.path.join(out_imgs, 'ImprintedKNET', data_point["filename"]), out_img)

                    return [out_img, out_VG, frame_objs, colour_seq, locations, hsv_colors], y_pred, y_true, run_eval #Continue to next image

                new_preds = extract_spatial(weak_idx, frame_objs, VG_base=VG_data)

                if new_preds:

                    # Take the max w.r.t. semantic relatedness
                    wlabel, wscore = max(new_preds.items(), key=lambda x: x[1])
                    wlabel = reverse_map(wlabel)

                    print("Based on all nearby objects this is a %s" % wlabel)
                    print("With confidence %f" % wscore)
                    print("Rankings after conceptnet relatedness:")
                    print(sorted(new_preds.items(), key=lambda x: x[1], reverse=True))

                    if sem == 'full':
                        # is the rel VG-validated w.r.t. floor?
                        orig_floor = proxy_floor(orig_label, VG_data['on'])
                        corrected_floor_out = proxy_floor(wlabel, VG_data['on'])

                        # ONLY if so (or if originally OOV) change it
                        if orig_floor == 'no synset' or orig_floor == corrected_floor_out:

                            print("Accepting proposed correction")
                            frame_objs[weak_idx] = (wlabel.replace('_', '-'), wscore, coords, new_preds)
                        else:

                            print("Rejecting suggested correction: replacement does not make sense w.r.t floor")


                    for lb, cf, (x, y, w, h), rank in frame_objs:

                        color = COLORS[all_classes.index(lb)]
                        cv2.rectangle(out_VG, (x, y), (x + w, y + h), color, 2)

                        if y - 10 > 0:
                            cv2.putText(out_VG, lb, (x - 10, y - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        else:
                            cv2.putText(out_VG, lb, (x - 10, y + h + 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    """
                    cv2.imwrite(os.path.join(out_imgs, 'Kground_ImprintedKNET', data_point["filename"]), out_VG)
                    """
            corr_preds, _, _, _ = zip(*frame_objs)
            y_pred.extend(corr_preds)


    else:

        print("No bboxes above threshold found by segmentation module")

    print("Took %f seconds " % (time.time() - whole_start))
    print("%EOF---------------------------------------------------------------------% \n")
    # cv2.imshow('union', out_img)
    # cv2.waitKey(10000)
    # cv2.destroyAllWindows()

    # cv2.imwrite(os.path.join(out_imgs, 'ImprintedKNET', data_point["filename"]), out_img)
    return [out_img, out_VG, frame_objs, colour_seq, locations, hsv_colors], y_pred, y_true, run_eval
