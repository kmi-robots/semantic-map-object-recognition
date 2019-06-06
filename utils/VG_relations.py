
import os
import json
import argparse
import time
from collections import Counter

"""
Can be used to create a spatially-centered 
copy of relations in Visual Genome

https://visualgenome.org/api/v0/api_home.html

"""


def parse_aliases(base, path_to_aliases):
    alias_dict = {}

    with open(os.path.join(base, path_to_aliases), 'r') as alsfile:

        lines = alsfile.read().split('\n')

    # print(lines)

    for line in lines:

        if line.split(',')[0] != '' and line.split(',')[0] != ' ':

            if line.split(',')[0] in alias_dict:

                alias_dict[line.split(',')[0]].extend(line.split(',')[1:])

            else:

                alias_dict[line.split(',')[0]] = line.split(',')[1:]

    # remove duplicates
    for key in alias_dict.keys():
        curr_list = alias_dict[key]
        alias_dict[key] = list(set(curr_list))


    return alias_dict

"""
def wntoimagenet(syns):

    \"""
    Converts input WordNet synset id to the
    format used in ImageNet and ShapeNet
    \"""
    offset = syns[0].offset()

    return "n{:08d}".format(offset)
"""

def filter_relations(base, path_to_json, aliases={}):

    filtered_dict = {}

    with open(os.path.join(base, path_to_json), 'r') as jsrel:

        full_data = json.load(jsrel)


    for dict_entry in full_data:

        rels = dict_entry["relationships"]

        for node in rels:

            core = node["predicate"].lower()

            sub = node["subject"]["synsets"]
            obj = node["object"]["synsets"]


            if len(sub) > 1 or len(obj)>1 or  core == '' or core==' ':

                #Skipping composite periods for now
                # ( e.g.  subject: green trees seen  pred: green trees by road object: trees on roadside. )
                #as well as ambiguous/empty predicates
                continue


            if core not in filtered_dict :

                filtered_dict[core]={}
                #if nor the predicate nor its aliases are already present
                filtered_dict[core]["relations"] = Counter()


            try:
                filtered_dict[core]["relations"][str((sub[0], obj[0]))] +=1


            except IndexError:

                #We are also skipping predicates that do not have both subject and
                #objects since now we are focusing on linking pairs of objects together
                continue

            """
            #Also add all aliases related to that predicate, if any
            if core in aliases:

                filtered_dict[core]["aliases"].extend(aliases[core])
                continue 
                
                
            for key, val in aliases.items():

                if core in val:

                    
                    
                    break

            """

    efficient_dict={}

    for core, node in filtered_dict.items():

        if core in aliases.keys():

            alters = aliases[core]

            efficient_dict[core] = {}

            efficient_dict[core]["aliases"] = alters

            efficient_dict[core]["relations"] = filtered_dict[core]["relations"]

            for al in alters:

                if al in filtered_dict.keys():

                    efficient_dict[core]["relations"] = efficient_dict[core]["relations"]+ filtered_dict[al]["relations"]


    print(len(filtered_dict['on']["relations"]))
    print("-------------------------------------------")
    print(len(efficient_dict['on']["relations"]))

    return efficient_dict

def main(path_to_json, path_to_aliases, out_path):

    base = os.getcwd()

    zero = time.time()

    print("Parsing aliases")

    alias_dict = parse_aliases(base, path_to_aliases)

    t1 = time.time()
    print('Took %f seconds' % (t1 - zero))

    #pretty-print resulting dictionary
    #print(json.dumps(alias_dict, indent=2))

    print('Processing all relations in Visual Genome')

    filtered_rels = filter_relations(base, path_to_json, aliases=alias_dict)

    print('Took %f seconds' % (time.time() - t1))

    with open(os.path.join(base, out_path), 'w') as outj:

        json.dump(filtered_rels, outj)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('reljson', help='Relative path to input json on VG relations')
    parser.add_argument('al', help='Relative path to input relationship aliases')
    parser.add_argument('out', help='Relative path to output JSON to be generated')
    args = parser.parse_args()


    main(args.reljson, args.al, args.out)