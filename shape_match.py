import os
import time
import cv2
import argparse 
from PIL import Image
import numpy as np
import sys
#from shapenet_select import findAndMatch
from pylab import arange, array, uint8 
from matplotlib import pyplot as plt
import rosbag
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
    #matdb = sio.loadmat(path_to_mat)    
    f = h5py.File(path_to_mat,'r')
    variables = f.items()

    for var in variables:

        #print(var)
        name = var[0]
        data = var[1]
        #print ("Name ", name)  # Name

        if type(data) is h5py.Dataset and name=='images':

            #print(data.shape)
            #data.resize((480,640,3,1449))
            #print(data.shape)
            # If DataSet pull the associated Data
            # If not a dataset, you may need to access the element sub-items
            #print(data.value)

    
            return data.value
            

if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument("imgpath", help="Provide path to the images to be classified") 
    parser.add_argument("modelpath", help="Provide path to the reference models")
    parser.add_argument("outpath", help="Provide path to output the similarity scores")
    args =parser.parse_args()

    start = time.time()
    
    #Imgs are read from the BYUDepth data set here
    img_mat = loadNYU(args.imgpath)

    img = img_mat[0]
    #img = np.reshape(img, (480,640,3))
    
    #print(img)
    #print(img.T.shape)
    #print(type(img))

    

    #cv2.imshow('Image', img)    
    cv2.imwrite('./test.png', img.T)   


    

    #shapeMatch(args.modelpath)        
 
    logging.info("Complete...took %f seconds" % float(time.time() - start))
