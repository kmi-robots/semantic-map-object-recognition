import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

def get_association_rules(input_dict):

    #TODO convert input dict into list of lists expected by Transaction Encoder
    #merge the frame level and make it a list of bboxes properties
    #also flatten the list of near objet to atomic values, i.e., from near: [a, b] to near-a, near-b

    all_boxes=[]
    for _, box_array in input_dict.items():

        all_boxes.extend(box_array)

    data = []

    for bbox_node in all_boxes:

        label= bbox_node["label"]   #object class
        vert_loc = bbox_node["frame_location"] #top/bottom/middle
        size = bbox_node["relative_size"]
        colour = bbox_node["dominant_colour"]

        try:

            dis = bbox_node["relative_distance"]

            data.append([label,vert_loc,size, dis, colour]+["near "+l for l in bbox_node["near"]])

        except KeyError:

            #dis can be None
            data.append([label, vert_loc, size, colour] + ["near " + l for l in bbox_node["near"]])


    te = TransactionEncoder()
    te_ary = te.fit(data).transform(data)

    df = pd.DataFrame(te_ary, columns=te.columns_)

    frequent_itemsets = apriori(df, min_support=0.0, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.0)

    #TODO convert df back to dictionary

    return rules