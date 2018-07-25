import os
import requests
import json
import sys
#from depth_segmentation import preproc
import cv2
import numpy as np


def findAndMatch(objs=os.listdir('/mnt/c/Users/HP/Desktop/KMI/shapenet-object-proof/merged'), comparout='/mnt/c/Users/HP/Desktop/KMI/test', contours2=None, img2=None, fname=None):

    simdict ={}    
    
    simdict["img_id"] = fname
    simdict['comparisons']=[]
    glob_min =1000.0

    for o in objs:

        img = cv2.imread(os.path.join(input_p,o), cv2.IMREAD_UNCHANGED)
        #edges = preproc(img)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 105)

        kernel = np.ones((5,5),np.uint8)
        dilation = cv2.dilate(edges,kernel,iterations = 1)

        #write partial output
        #Only external contours instead of full hierarchy here 
        __, contours,hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        imgo = cv2.drawContours(dilation, contours, -1, (0,255,0), 3)

        #mask = np.zeros(imgo.shape[:2], dtype="uint8") * 255
        dil = cv2.dilate(imgo, kernel, iterations=1)
        #cv2.drawContours(mask, contours, -1, (255, 255, 255), -1)
        #imgm = cv2.bitwise_and(imgo, imgo, mask=mask)

        cv2.imwrite(os.path.join(output_p, o), dil)
        #Contour detection adds too much noise here
        #conts = contour_det(edges)
        #draw_cnt(conts, img, o, output_p)

        if contours2 is None or img2 is None:
            continue

        jet = cv2.applyColorMap(img2,cv2.DIST_L2,5) 
        


        comparison={}  
        comparison["compared_obj"] = o[:-4]
        comparison["similarities"] =[]

        i=1

        for cnt2 in contours2:
         
          
            cv2.drawContours(jet, contours2[0], -1, (0, 0, 255), 3)
            #cv2.drawContours(img2, contours2[1], -1, (255, 0, 0), 3)
            #cv2.drawContours(img2, contours2[2], -1, (0, 255, 0), 3)
            #print(len(contours2))   
            
            cv2.imwrite(os.path.join(comparout,fname+"_"+str(i)+'.png'), jet)
    
            
            '''
            if len(contours) >1 :
                print("Too many contours in object %s!!" % str(o))
                sys.exit(0)
            '''

            #for cnt in contours:
                
            cv2.drawContours(img, contours[0], -1, (0, 0, 255), 3)

            cv2.imwrite(os.path.join(comparout,o[:-4]+"_"+str(i)+o[-4:]), img)

                 
            print("-----Comparison no. %i ----------" % i)    
            print(cv2.matchShapes(cnt2, contours[0],1,0.0))
            
             
            comparison["similarities"].append((i, cv2.matchShapes(cnt2, contours[0],1,0.0)))

            i+=1
          
        comparison["similarities"] = sorted(comparison["similarities"],key=lambda x:(x[1],x[0]))

        iterat, curr_min = min(comparison["similarities"], key = lambda t: t[1])  
        #print(curr_min) 
        #print(type(curr_min))
        #sys.exit(0)
      
        if curr_min < glob_min:
            glob_min = curr_min
            obj_min = o

        simdict['comparisons'].append(comparison)


    #Output most similar object 
    simdict["min"]=(o, glob_min)

    return simdict 
      
        
SOLR_BASE='https://www.shapenet.org/solr/models3d/'
n_rows = 1000
out_format = 'json'
input_p ='/mnt/c/Users/HP/Desktop/KMI/shapenet-object-proof/merged'
output_p= '/mnt/c/Users/HP/Desktop/KMI/shapenet-object-proof/merged-out'

if __name__ == '__main__':


    objects = os.listdir(input_p)
    findAndMatch(objects)

'''
#Hardcoded keyword list for object models
#TODO: change to argparse 
queries =['chair', 'mug', 'monitor', 'plant']

objs = os.listdir(input_p)

results=[]
ids=[]
full_ids=[]


    

#Querying Shapenet Solr Engine, given a list of keywords
for q in queries:

    data = requests.get(SOLR_BASE+'select?q='+q+'AND+source%3A3dw&rows='+str(n_rows)+'&wt='+out_format)
    
    records = data.json()["response"]["docs"]
    results.append(records)

    ids.extend([record["id"] for record in records])
    
    full_ids.extend([record["fullId"] for record in records])
     
     
#print(len(results))
#print(results[2])
#print(ids)
#print(len(ids))
'''
#Use the same id to retrieve image


#Import openCV segmentation from other script

    #Save results for later matching


