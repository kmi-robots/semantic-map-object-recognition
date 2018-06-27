from __future__ import division
import visual_genome.utils as utils
import json 
import time
import visual_genome.local as vglocal
from nltk.stem import WordNetLemmatizer
import os
import sys
import argparse
import shutil
from PIL import Image as PIL_Image

def parse_obj(obj_name):

    try:
   
        objectn= ' '.join(obj_name.split('_')) 
 
    except:

        objectn = obj_name
    return objectn

    
def load_dict(path_to_concepts): 

    with open(path_to_concepts) as conceptf:

        concepts= json.load(conceptf)
        #print(type(concepts))


    all_concepts=[]

    for key, node in concepts.items():
        
        #print(node)
        #sys.exit(0)
        obj_list = node["objects"]
        obj_list= [parse_obj(obj) for obj in obj_list]
        all_concepts.extend(obj_list)   

    return all_concepts
    #print(all_concepts)
    

def extract_graphs(basep, all_concepts): 

    print("Starting to collect graphs...")

    all_objs=[]

    lemmatizer = WordNetLemmatizer()

    out_path= os.path.join(os.getcwd(),'data/out')


    #Hardcoded, based on VG total size
    for i in range(1,108000, 100):

        START= i
        END= START +100
    
        graphs = vglocal.get_scene_graphs(start_index=START, end_index=END, data_dir='data/', image_data_dir=basep)    

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

            with open( os.path.join(out_path,str(graph.image.id)+'.txt'), 'w', encoding='utf-8') as jout:
                
                for obj in diction:

                    x= obj["x"]
                    y= obj["y"]
                    width = obj["w"]
                    height = obj["h"]
                
                    #Dump results in "YOLO-like" format
                    jout.write(str(obj["synsets"]) + " "+ str(x) +" "+ str(y) + " "+ str(width) + " "+ str(height)+"\n")     
            


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

    #Remove dups    
    u_objs=list(set(all_objs))

    print(len(u_objs))
    #print(keep)

    return u_objs


def check_dictionary(onlist, vocabulary):

    classnos=[]

    for term in onlist:

        term= term.replace("]","").replace("[","")
        if term not in vocabulary: 

            vocabulary[term] =len(vocabulary) 
               
        classnos.append(vocabulary[term])


    return classnos


def convert(clno, imgw, imgh, xmin, xmax, ymin, ymax, w, h):

    #Converts (x,y) -top left corner, width and height (taken from image resolution)
    #to YOLO format, i.e., center in X, center in Y, width in X, width in Y

    #max for YOLO is 416x416
    yolo_width= 416
    yolo_height= 416   
 
    



    dw= 1./int(imgw)
    dh = 1./int(imgh)

    

    x_center = (xmin + xmax)/2.0
    y_center = (ymin + ymax)/2.0         


    x_center = str(x_center*dw)
    y_center = str(y_center*dh)
    h = str(h*dh)
    w = str(w*dw)
    clno= str(clno)

    return " ".join([clno, x_center, y_center, w, h])


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("concepts", help="Path to concepts_larger.json", default='data')
    parser.add_argument("-slicepath", default='data/by-id/', help='path to individual json files containing scene graphs')
    parser.add_argument("--slice", required=False, action='store_true', default=False, help='Run with this flag =True if scene_graphs.json has not be sliced by id yet')

    parser.add_argument("--extract", required=False, action='store_true', default=False, help='Extracts bounding boxes from scene graphs, after filtering image set by ConceptNet keywords')

    parser.add_argument("--map", required=False, action='store_true', default=False, help='to enable mapping from object name to class no.')
    parser.add_argument("--neww", required=False, default=416, help='new width of resized images')
    
    parser.add_argument("--newh", required=False, default=416, help='new height of resized images')
    
    args =parser.parse_args()
    
    start = time.time()
    
    path_to_concepts= os.path.join(args.concepts,'concepts_larger.json')
    basep =args.slicepath  
    
    outdir= os.path.join(os.getcwd(),'data/out')
    backup = os.path.join(os.getcwd(),'data/out-formatted')
    

    if not os.path.isdir(basep):
        os.mkdir(basep)    


    if not os.path.isdir(outdir):

        os.mkdir(outdir)
    
    if not os.path.isdir(backup):

        os.mkdir(backup)
    
    if args.slice:       

        #Divides up scene graphs in separate json files by image id
        vglocal.save_scene_graphs_by_id(data_dir=basep, image_data_dir=basep)


    #Load keywords from concepts_larger.json
    concept_list= load_dict(path_to_concepts)

    if args.extract:
        #Extracts range of scene_graphs, given data dir
        #Requires scene_graphs.json to be sliced by image id beforehand

        #Get all scene graphs
        unique_objects = extract_graphs(basep, concept_list)

        #Write extracted objects locally for later
        with open(os.path.join(os.getcwd(),'class_list.txt'), 'w') as outf:
    
           [outf.write(str(obj)+'\n') for obj in unique_objects]


    if args.map:
        
        #Find unique object list (if available) and parse from names to classno.
        outfiles = os.listdir(outdir)
        
        less_objects=[] 
        all_synsets=[]
        object_voc={}

        for filen in outfiles:

            #print(filen)
            templ=[]

            with open(os.path.join(outdir,filen), 'r') as bboxf:
        
                bboxes = bboxf.readlines()
               
                for line in bboxes:
   
                    if line[:2] =='[]':

                        continue
                    
                    #print(line) 
                    stringp= str(line).replace('"', "'").split("['")[1]
                    stringp = stringp.split("'] ")[0].replace(" ","")
                    stringp= stringp.replace("'", "").split(",")
                    templ.extend(stringp)
                    #sys.exit(0)    
          
                templ= list(set(templ))
                 
                all_synsets.extend(templ)

                listno = check_dictionary(templ, object_voc) 
            
            
            with open(os.path.join(backup, filen), 'w') as cleanf:
            
                 
                for line in bboxes:

                    if line[:2] =='[]':

                        continue
                        #print(line)
                    #print(line) 
                    #sys.exit(0)
                    tobr= line.replace('"',"'").split("'] ")[0]
                    key = tobr.replace('"',"'").split("['")[1]

                    try:
                        #If more words, take the first one
                        tgt = key.replace("'","").split(",")[0].replace(" ","")

                    except:
                        #If it is just one word 
                        tgt = key.replace("'","").replace(" ","")

                    #print(tgt) 
                    linel= line.replace('"',"'").replace(tobr+"']", str(object_voc[tgt.replace("[","").replace("]","")])).split(" ")
                    

                    #Resolution is different from image to image
                    try:
                        img = PIL_Image.open(open('/home/linuxadmin/visual_genome_python_driver/VG_100K/'+filen[:-3]+'jpg', 'r+b'))
                        
                    except FileNotFoundError:
                        
                        img= PIL_Image.open(open('/home/linuxadmin/visual_genome_python_driver/VG_100K_2/'+filen[:-3]+'jpg', 'r+b'))

                    img_w, img_h = img.size                 
  
                    #print(line)
                    classno= linel[0]
                    x_tl = float(linel[1])
                    y_tl = float(linel[2])
                    w = float(linel[3])
                    h = float(linel[4])
                   
                    #Adapt bounding boxes after resizing 
                    new_iw = args.neww
                    new_ih = args.newh

                    x_tl = float(x_tl/img_w)*new_iw
                    y_tl = float(y_tl/img_h)*new_ih
                    w = float(w/img_w)*new_iw
                    h = float(h/img_h)*new_ih
                    
                    xmin= x_tl
                    xmax= x_tl + w
                    ymin = y_tl - h
                    ymax = y_tl

                    linec= convert(str(classno), new_iw, new_ih, xmin, xmax, ymin,ymax, w, h)

                    cleanf.write(linec+"\n")

            #sys.exit(0)
        
        print("Done formatting - Darknet will have to be setup on %i classes" % len(object_voc))
        #less_objects= list(set(less_objects))
        #obj_d = map_objects(less_objects)
                

    all_synsets = list(set(all_synsets))

    with open('data/class_list.txt', 'w') as cfile:

        [cfile.write(str(synset)+"\n") for synset in all_synsets]
    

    print("Complete...took %f seconds" % float(time.time() - start))


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

