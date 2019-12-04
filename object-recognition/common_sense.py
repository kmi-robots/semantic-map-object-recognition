
from nltk.corpus import wordnet
from pattern.en import singularize, pluralize
import requests
import time
from collections import Counter, OrderedDict
import json



def compute_sem_sim(wemb1, wemb2):
    """
    Returns semantic similarity between two word embeddings
    based on Numberbatch 17.06 mini
    """

    dotprod = (wemb1 * wemb2).sum()
    sq_wemb1 = wemb1.pow(2).sum()
    sq_wemb2 = wemb2.pow(2).sum()

    return (pow(sq_wemb1, 0.5) * pow(sq_wemb1, 0.5)) / dotprod


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

        label = "power-cables"

    elif label == "computers":

        label = "desktop-pcs"

    elif label == "computer_monitors":

        label = "monitors"

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

    return label, get_synset(wn_label)  # [:-1]  # remove plurals


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

    # numberbatch_vecs = pandas.read_hdf(path_to_dict.split("KMi_conceptrel.json")[0]+'mini.h5')
    base = '/c/en/'

    score_dict = {}

    for term in class_set:

        print(term)

        # all other classes but itself

        term_dict = {}

        # emb1 = numberbatch_vecs.loc[base + term]

        for term2 in [c for c in class_set if c != term]:  # Skip identity case

            try:

                node = requests.get('http://api.conceptnet.io/relatedness?node1=' + \
                                    base + term + '&node2=' + base + term2).json()

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

        # Relatedness scores for that term, in ascending order
        term_dict = sorted(term_dict.items(), key=lambda kv: kv[1])  # , reverse=True)
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

    # print(base_score)
    # Is there any other label in the ranking making more sense?

    for o_class, confidence in candidates.items():

        f_class, _ = formatlabel(o_class)

        if f_class == subject:
            continue  # Skip the object itself

        score = call_cp_api(f_class, object)

        if score > base_score:
            base_score = score
            pred_subject = o_class

    print("CONCEPTNET: Within the ranking, the most likely subject is %s" % pred_subject)
    if singularize(pred_subject) == pred_subject:
        # Re-format back for evaluation
        pred_subject = pluralize(pred_subject)

    pred_subject = reverse_map(pred_subject)

    return pred_subject.replace('_', '-'), base_score




def extract_spatial(weak_idx, obj_list, VG_base=None):
    preds, confs, coords, rankings = zip(*obj_list)

    weak_pred, weak_synset = formatlabel(preds[weak_idx])

    # weak_conf = confs[weak_idx]
    x_a = coords[weak_idx][0]
    y_a = coords[weak_idx][1]
    w_a = coords[weak_idx][2]
    h_a = coords[weak_idx][3]
    weak_rank = rankings[weak_idx]

    new_ranking = Counter()

    # rel_graph["subject"] = weak_pred
    # rel_graph["synset"] = weak_synset

    # Are any other objects are nearby?
    for label, score, bbox, ranking in [obj_list[i] for i in range(len(obj_list)) if i != weak_idx]:

        # Add threshold: only link with object we are reasonably confident about
        if score >= 1.14:  # 2.50: more than 50% confident about the other object

            x_b = bbox[0]
            y_b = bbox[1]
            w_b = bbox[2]
            h_b = bbox[3]

            # Spatial ref is relative to anchor object
            spatial_rel = check_spatial(x_a, x_b, y_a, y_b, w_a, h_a, w_b, h_b)

            if spatial_rel is not None:

                sem_obj, obj_syn = formatlabel(label)

                # Verify on supporting knowledge if found relation makes sense

                if weak_synset and obj_syn:

                    print(
                        weak_pred + "( " + weak_synset.name() + " ) " + spatial_rel + " " + sem_obj + "( " + obj_syn.name() + " ) \n")

                else:

                    print(weak_pred + " " + spatial_rel + " " + sem_obj + "\n")

                # Does this relation make sense?
                # 1) In conceptnet?
                cp_pred, cp_score = conceptnet_relatedness(weak_pred, weak_rank, sem_obj)

                # Changed: no first filter by max

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
TABLE = wordnet.synsets("table")[1].name()  # table.n.01 is a data table



def extract_spatial_unbound(obj_idx, obj_list, SR_KB):

    #same as extract_spatial but without:
    # thresholding on confidence
    # validation against external knowledge
    # ranking correction
    # other differences:
    # requires counter of prior relations found
    # includes rel with floor

    preds, confs, coords, rankings = zip(*obj_list)

    orig_alias = preds[obj_idx]
    obj_pred, obj_synset = formatlabel(orig_alias)

    x_a, y_a, w_a, h_a = coords[obj_idx]

    floor_rel = check_horizontal_unbound(y_a, h_a)
    SR_KB[(floor_rel, (obj_pred, obj_synset), ("floor", FLOOR))] += 1

    # Are any other objects are nearby?
    for label, score, (x_b,y_b,w_b,h_b), ranking in [obj_list[i] for i in range(len(obj_list)) if i != obj_idx]:

            # Spatial ref is relative to anchor object
            spatial_rel = check_spatial(x_a, x_b, y_a, y_b, w_a, h_a, w_b, h_b)

            if spatial_rel is not None:

                sem_obj, obj_syn = formatlabel(label)

                SR_KB[(spatial_rel, (obj_pred, obj_synset),(sem_obj,obj_syn))] += 1

    return SR_KB


def check_horizontal_unbound(y, h, img_res=(640, 480)):
    bar_y = y + h  # lower y = base of bbox
    thresh = img_res[1] - img_res[1]/4

    if bar_y > thresh: #and bar_y <= img_res[1]:

        return "on "

    else:

        return "above"


# While the piece of furniture is table.n.02


def check_horizontal(y, h, img_res=(640, 480)):
    bar_y = y + h  # /2  #Replace bar_y with actual lower y
    thresh = img_res[1] - img_res[1] / 3

    if bar_y > thresh and bar_y <= img_res[1]:

        return "on " + FLOOR

    else:

        return None


def check_spatial(x1, x2, y1, y2, w1, h1, w2, h2):
    # CHANGED: make horizontal bbox expansion relative to each bbox size as opposed to absolute px value
    lowto = 10
    hito = w1

    bar_x2 = x2 + w2 / 2
    bar_y2 = y2 + h2 / 2

    if bar_y2 > y1 + h1 and bar_x2 >= x1 - hito and bar_x2 <= x1 + w1 + hito:

        return "on"

    elif bar_y2 < y1 and bar_x2 >= x1 - hito and bar_x2 <= x1 + w1 + hito:

        return "under"

    elif bar_x2 <= x1 + w1 + hito and bar_x2 >= x1 - hito and bar_y2 <= y1 + h1 + lowto and bar_y2 >= y1 - lowto:

        return "near"

    else:

        return None


def replace(ranking, coords, VG_onrel, frame_objs, weak_idx, replaced=False):

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

                if (on_floor > on_table + 80):

                    return "on " + FLOOR

                else:

                    return "both"

            except KeyError:

                return "on " + FLOOR

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

        # not applicable to Health and Safety objects or other OOV

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

            # Does not have on floor rels in VG
            print("Not doing anything ")
            pass

    return frame_objs, rflag


def show_leastconf(scene_objs):

    preds, confs, coords, rankings = zip(*scene_objs)

    if min(confs) < 1.14:  # 2.5:  #less than 50%

        i = confs.index(min(confs))

        print("Weakest prediction was \n")
        print(preds[i])

        print("With associated ranking scores \n")
        print(rankings[i])


        return i

    else:

        return None