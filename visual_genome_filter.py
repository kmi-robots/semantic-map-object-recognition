import visual_genome.utils as utils
import json 
import time
import visual_genome.local as vglocal
from nltk.stem import WordNetLemmatizer
import os
import sys

def parse_obj(obj_name):

    try:
   
        objectn= ' '.join(obj_name.split('_')) 
 
    except:

        objectn = obj_name
    return objectn

####HARDCODED###################################
path_to_concepts= 'data/concepts_larger.json'
################################################

print("Start")
start = time.time()

#Already done do not uncomment in this environment! 
#Divides up scene graphs in separate json files by image id
#vglocal.save_scene_graphs_by_id(data_dir='data/', image_data_dir='data/by-id/')

#Extracts range of scene_graphs, given data dir
#Requires scene_graphs.json to be sliced by image id beforehand

#Get all scene graphs

print("Starting to collect graphs...")

#graphs = vglocal.get_scene_graphs(start_index=0, end_index=END, data_dir='data/', image_data_dir='data/by-id/')

basep = 'data/by-id'
#filenames= os.listdir(basep)


print("All graphs loaded, took %f seconds" % float(time.time() - start)) 

all_objs=[]

lemmatizer = WordNetLemmatizer()

with open(path_to_concepts) as conceptf:

    concepts= json.load(conceptf)
    print(type(concepts))

all_concepts=[]

for key, node in concepts.items():
        
    print(node)
    #sys.exit(0)
    obj_list = node["objects"]
    obj_list= [parse_obj(obj) for obj in obj_list]
    all_concepts.extend(obj_list)   


print(all_concepts)
#keep=[]

for i in range(1,108000, 100):

    START= i
    END= START +100
    
    graphs = vglocal.get_scene_graphs(start_index=START, end_index=END, data_dir='data/', image_data_dir='data/by-id/')    

    print("------First %i graphs loaded---------------" % END)    
    n=0

    for graph in graphs:
        
        overlap = [str(obj) for obj in graph.objects if str(obj) in all_concepts]
        
        #If less than six objects from the vocabulary are included, then skip image
        if len(overlap)<5:
            continue 

        #Otherwise add image metadata to keep list
        #keep.append(graph.image)

        current_n= str(graph.image.id)+'.json'     
        with open(os.path.join(basep, current_n)) as jin:
            
            diction =json.load(jin)["objects"]
      
        '''  
        xs=[(obj["name"], obj["x"]) for obj in diction]
        ys=[(obj["name"], obj["y"]) for obj in diction]
        widths=[(obj["name"], obj["w"]) for obj in diction]
        heights=[(obj["name"], obj["h"]) for obj in diction]
        '''

        with open(os.path.join('/home/linuxadmin/visual_genome_python_driver/data/out', str(graph.image.id)+'.txt'), 'w', encoding='utf-8') as jout:
            for obj in diction:

                x= obj["x"]
                y= obj["y"]
                width = obj["w"]
                height = obj["h"]
                
                #Dump results in YOLO-like format
                jout.write(str(obj["names"]) + " "+ str(x) +" "+ str(y) + " "+ str(width) + " "+ str(height)+"\n")     

        #sys.exit(0)

        for o in graph.objects:

            if len(str(o).split())> 1:

                o = ' '.join([lemmatizer.lemmatize(w) for w in str(o).split()])
            
            else:

                o= lemmatizer.lemmatize(str(o)) 

            all_objs.append(o)
        

        n+=1
                        
        if n % 10 == 0:

            print("Classes extracted from %i graphs" % n)

    
    print("%i objects found so far" % len(all_objs))
    #print("%i images will be kept" % len(keep))
    #break

#Remove dups    
u_objs=list(set(all_objs))

print(len(u_objs))
#print(keep)

with open('/home/linuxadmin/visual_genome_python_driver/class_list.txt', 'w') as outf:
    
    [outf.write(str(obj)+'\n') for obj in u_objs]

#images ={}
#images["image_list"]= keep
#with open('be_kept.txt', 'w', encoding='utf-8') as klist:
    
#    [klist.write(str(keepimg)+'\n') for keepimg in keep] 

#print(u_objs)

#graph = vglocal.get_scene_graph(1, images='data/', image_data_dir='data/by-id/',synset_file='data/synsets.json')

#print(graph)


#Load json with target concepts

'''
with open('/concepts_larger.json') as injson:
  
    cbase = json.load(insjon.read())

#Lemmatize concepts


#Use those to filter VG 

'''

print("Complete...took %f seconds" % float(time.time() - start))
