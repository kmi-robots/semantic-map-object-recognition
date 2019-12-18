import rule_extractor as rex
from collections import OrderedDict
import os
import json


if __name__ == '__main__':


    with open(os.getcwd()+'/SR_KB.json','r') as file_in:

        input_dictionary = json.load(file_in)

        print("Starting rule mining from collected observations...")

        rule_df = rex.get_association_rules(input_dictionary["global_rels"])

        rule_df.to_csv(os.getcwd()+'/data/kmi_test_activity_rules.csv')
        # save locally
        print("Saving rules locally...")
        rule_df.to_pickle(os.getcwd() + '/data/KMi_NW_activity_test_extracted_rules.pkl')
        print("Pickled DataFrame saved under ../data/extracted_rules.pkl")

