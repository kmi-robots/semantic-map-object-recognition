import rosbag
#import matlab
#from mat_segmenting import extract_masks
#import matlab.engine
import numpy as np
import os
import time
import cv2
import argparse 
from PIL import Image
import sys
#from shapenet_select import findAndMatch
from pylab import arange, array, uint8 
from matplotlib import pyplot as plt
import subprocess, yaml
from cv_bridge import CvBridge, CvBridgeError
import json
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import random as rng
import scipy.io as sio
import logging
import h5py



def shapeMatch(shape1, model):


    ret, thresh2 = cv2.threshold(model, 127, 255,1)
    _, contours,hierarchy = cv2.findContours(thresh2,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    

    areas = [cv2.contourArea(cnt) for cnt in contours]
   
    idx = areas.index(max(areas))    

    shape2 = contours[idx]

    model  = cv2.cvtColor(model,cv2.COLOR_GRAY2RGB)
    #cv2.drawContours(model, [shape2], 0, (0,255,0), 3)
    
    #cv2.imshow('', model)
    #cv2.waitKey(8000)
    #sys.exit(0)

    return cv2.matchShapes(shape1, shape2,1,0.0)
    


def loadNYU(path_to_mat):

    '''
    Loading the NYUDepth label data set  
    from a Matlab file
    '''
    print('Loading Matlab data set...') 

    
    #matdb = sio.loadmat(path_to_mat)    
    f = h5py.File(path_to_mat,'r')

    #print(f.keys())
    ''' 
    label_names = f['names']  
 
    for i, st in  enumerate(label_names[0]):
        #print(st)
        #print(type(st))
        obj= f[st]

        str1 = ''.join(chr(i) for i in obj[:])

        #print(str1)
        #sys.exit(0)

        if str1 =='chair': 

            label_index=i 
            print('Hey, I have just found a chair!')
   
    #print(label_names)
    
   
    for lab in label_names.value:
        
        print(lab) 
        print(f[lab])
   
    '''
    
    variables = f.items()

    
    for var in variables:

        #print(var)
        name = var[0]
        data = var[1]
        #print ("Name ", name)  # Name

        '''
        if type(data) is h5py.Dataset and name=='namesToIds':

            print(data.keys())
            label_names = data.value
        
          
        if type(data) is h5py.Dataset and name=='instances':

            #Use Matlab script from NYUDepth's Toolbox 
            #to extract masks related to each instance in the scene            
            instances = data.value            #eng = matlab.engine.start_matlab()
            #eng.get_instance_masks(instances, nargout=0)   #nargout=0 to avoid output printing within Python

        elif type(data) is h5py.Dataset and name=='labels':

            labels =data.value
        '''

        if type(data) is h5py.Dataset and name=='images':

            #print(data.shape)
            #data.resize((480,640,3,1449))
            #print(data.shape)
            # If DataSet pull the associated Data
            # If not a dataset, you may need to access the element sub-items
            return data.value


        
    '''   
    print(label_names)
    print('Label names collected')
    
    return label_index
    '''

        
if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument("imgpath", help="Provide path to the images to be classified") 
    parser.add_argument("modelpath", help="Provide path to the reference models")
    #parser.add_argument("pathtomasks", help="Provide path to the segmented objects to be evaluated")
    parser.add_argument("outpath", help="Provide path to output the similarity scores")
    args =parser.parse_args()

    start = time.time()
   
     
    #CHANGED: Ended up implementing dataset pre-proc in Matlab directly
    #see maskfiles.m

    #Imgs are read from the BYUDepth data set here
    #images = loadNYU(args.imgpath).T    #We have to take the transpose here from what it returns 

    #print(images.shape)
    
    modelfiles = os.listdir(args.modelpath)
    modelpaths = [os.path.join(args.modelpath, mfile) for mfile in modelfiles]

    objectn = os.listdir(args.imgpath)
    objectpaths = [os.path.join(args.imgpath, name) for name in objectn]

    for filep in objectpaths:

        simdict = {}
        l = filep.split("/")
        fname = l[len(l)-1]
        objcat = l[len(l)-2]

        #Read image in grayscale
        objimg = cv2.imread(filep, 0) 
    
        simdict["img_id"] = fname
        simdict['obj_category']= objcat
        simdict['comparisons']=[]
        glob_min =1000.0
        
        #Extract shape
        try:
            ret, thresh = cv2.threshold(objimg, 0, 255,0)
            _, contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


            areas = [cv2.contourArea(cnt) for cnt in contours]
   
            idx = areas.index(max(areas))    


            shape1 = contours[idx]

            img =  cv2.imread(filep, 1)  
            #cv2.drawContours(img, [shape1], 0, (0,0,255), 3)
            #cv2.imshow('', img)
            
            #cv2.waitKey(5000)

            #cv2.imshow('', thresh)
            #cv2.waitKey(2000)
        except Exception as e:

            #Empty masks 
            print('Problem while processing image %s' % fname)
            print(str(e))
            simdict['error'] ='Segmented area could not be processed'
            #Output dictionary as JSON file
            jname = fname[:-3]+'json'
            
            #cv2.imshow('', objimg)

            #cv2.waitKey(2000)
            #cv2.imshow('', thresh)
            #cv2.waitKey(2000)
            #print(contours)

            #cv2.drawContours(objimg, contours, 0, (0,255,0), 3)

            #cv2.imshow('', objimg)

            #cv2.waitKey(5000)
            
            with open(os.path.join(args.outpath, jname), 'w') as outf:

                json.dump(simdict, outf, indent=4)
            
            continue
 

        #Compare with all Shapenet models
        for modelp in modelpaths: 

            comparison={} 
            
            lm = modelp.split('/')             
            modname = lm[len(lm)-1]

            comparison["compared_obj"] = modname
            comparison["similarities"] =[]   

            mimage = cv2.imread(modelp, 0)
            
            #Do a pointwise comparison within each couple
            score= shapeMatch(shape1, mimage)        
            comparison["similarities"].append(score)
            
            #sys.exit(0)

            try:
                iterat, curr_min = min(comparison["similarities"], key = lambda t: t[1])  

            
            except TypeError:

                #Not enough values yet
                curr_min = score

            if curr_min < glob_min:
                glob_min = curr_min
                obj_min = modname

            simdict['comparisons'].append(comparison)

        #Sort by descending similarity
        #simdict["comparisons"] = sorted(simdict["comparisons"],key=lambda x:(x[1],x[0]))
        
        #Add key field for most similar object 
        simdict["min"]=(modname, glob_min)

        #Output dictionary as JSON file
        jname = fname[:-3]+'json'
        with open(os.path.join(args.outpath, jname), 'w') as outf:

            json.dump(simdict, outf, indent=4)

        #sys.exit(0)
 
    logging.info("Complete...took %f seconds" % float(time.time() - start))
