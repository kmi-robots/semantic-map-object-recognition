import rosbag
import matlab
from mat_segmenting import extract_masks
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



def shapeMatch(path_to_models):

    objs=os.listdir(path_to_models)
    simdict ={}    
    shape1= None
    shape2= None
    
    cv2.matchShapes(shape1, shape2,1,0.0)
    
    return simdict 


def loadNYU(path_to_mat):

    '''
    Loading the NYUDepth label data set  
    from a Matlab file
    '''
    print('Loading Matlab data set...') 

    
    #matdb = sio.loadmat(path_to_mat)    
    f = h5py.File(path_to_mat,'r')

    label_names = f['names']  
 
    for st in  label_names[0]:
        #print(st)
        #print(type(st))
        obj= f[st]

        str1 = ''.join(chr(i) for i in obj[:])

        if str1 =='chair': 
            print('Hey, I have just found a chair!')
   
    #print(label_names)

    '''
    for lab in label_names.value:
        
        print(lab) 
        print(f[lab])
   
    '''
    '''
    variables = f.items()

    
    for var in variables:

        #print(var)
        name = var[0]
        data = var[1]
        #print ("Name ", name)  # Name


        if type(data) is h5py.Dataset and name=='names':

            label_names = data.value
       
          
        if type(data) is h5py.Dataset and name=='instances':

            #Use Matlab script from NYUDepth's Toolbox 
            #to extract masks related to each instance in the scene            
            instances = data.value            #eng = matlab.engine.start_matlab()
            #eng.get_instance_masks(instances, nargout=0)   #nargout=0 to avoid output printing within Python

        elif type(data) is h5py.Dataset and name=='labels':

            labels =data.value
        
        if type(data) is h5py.Dataset and name=='images':

            #print(data.shape)
            #data.resize((480,640,3,1449))
            #print(data.shape)
            # If DataSet pull the associated Data
            # If not a dataset, you may need to access the element sub-items
            #print(data.value)

    
            return data.value
        
    '''    

    print('Label names collected')
    
    return label_names


def get_masks(path_to_masks):

    all_masks=[]
    fnames = os.listdir(path_to_masks)[:3]

    for filen in fnames:


        mat = matlab.engine.start_matlab()
        f = mat.load(os.path.join(path_to_masks, filen), nargout=1)

        m = f['masks']
        l = f['lab']
        all_masks.append((m,l))
        
        '''
        f = h5py.File(os.path.join(path_to_masks,filen),'r')

        variables = f.items()

    
        for var in variables:

            #print(var)
            name = var[0]
            data = var[1]

            if type(data) is h5py.Dataset and name=='masks':

                m = data.value
            if type(data) is h5py.Dataset and name=='lab':

                l = data.value

        all_masks.append((m,l))

        '''
    '''
    print('Extracting object labels and masks, for each image')
    
    print(len(zip(labels,instances))) 
    print(zip(labels,instances)[:1])   

    masks=[]
    for label_mat, instance_mat in zip(labels,instances):

        #print(label_mat.shape)
        #print(instance_mat.shape)
        m, l = extract_masks(matlab.uint8(label_mat.tolist()), matlab.uint8(instance_mat.tolist()))
                
        masks.append((m,l))
        sys.exit(0)

    #masks = [extract_masks(matlab.uint8(label_mat.tolist()), matlab.uint8(instance_mat.tolist())) for label_mat, instance_mat in zip(labels,instances)]
    

    for in_masks, in_labels in masks:

        print(in_labels)
        system.exit(0)
    '''
    return all_masks 
        
if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument("imgpath", help="Provide path to the images to be classified") 
    parser.add_argument("modelpath", help="Provide path to the reference models")
    parser.add_argument("pathtomasks", help="Provide path to the mat files containing masks and label info")
    parser.add_argument("outpath", help="Provide path to output the similarity scores")
    args =parser.parse_args()

    start = time.time()
   
   
    #logging.info('Off we go!')
 
    #Imgs are read from the BYUDepth data set here
    names = loadNYU(args.imgpath)
    #print(names)
    all_masks = get_masks(args.pathtomasks)
    print(len(all_masks))  
    
    '''
    img = img_mat[0]
    #img = np.reshape(img, (480,640,3))
    
    #print(img)
    #print(img.T.shape)
    #print(type(img))

    

    #cv2.imshow('Image', img)    
    cv2.imwrite('./test.png', img.T)   
    '''

    

    #shapeMatch(args.modelpath)        
 
    logging.info("Complete...took %f seconds" % float(time.time() - start))
