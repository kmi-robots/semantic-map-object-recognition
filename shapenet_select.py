import os
import requests
import json
import sys


SOLR_BASE='https://www.shapenet.org/solr/models3d/'
n_rows = 1000
out_format = 'json'

#Hardcoded keyword list for object models
#TODO: change to argparse 
queries =['chair', 'mug', 'monitor', 'plant']

results=[]
ids=[]
full_ids=[]

#Querying Shapenet Solr Engine, given a list of keywords
for q in queries:

    data = requests.get(SOLR_BASE+'select?q='+q+'AND+source%3A3dw&rows='+str(n_rows)+'&wt='+out_format)
    
    records = data.json()["response"]["docs"]
    results.append(records)

    ids.extend([record["id"] for record in records])
    
    full_ids.extend([record["fullid"] for record in records])
     
     
#print(len(results))
#print(results[2])
#print(ids)
#print(len(ids))

#Use the same id to retrieve image


#Import openCV segmentation from other script


#Save results for later matching



