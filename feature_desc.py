from __future__ import division
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
import csv
import statistics as stat
import random


#### A bunch of ugly hardcoded object IDs    ##############################

#Micro, i.e., same object different views
chair1=['9e14d77634cf619f174b6156db666192-0.png', '9e14d77634cf619f174b6156db666192-2.png', '9e14d77634cf619f174b6156db666192-5.png', '9e14d77634cf619f174b6156db666192-7.png', '9e14d77634cf619f174b6156db666192-10.png', '9e14d77634cf619f174b6156db666192-12.png', 'c.png']
chair2=['49918114029ce6a63db5e7f805103dd-0.png', '49918114029ce6a63db5e7f805103dd-1.png' ,'49918114029ce6a63db5e7f805103dd-5.png', '49918114029ce6a63db5e7f805103dd-6.png', '49918114029ce6a63db5e7f805103dd-9.png', '49918114029ce6a63db5e7f805103dd-11.png', '49918114029ce6a63db5e7f805103dd-13.png']

plant1=['4d637018815139ab97d540195229f372-1.png', '4d637018815139ab97d540195229f372-3.png', '4d637018815139ab97d540195229f372-7.png', '4d637018815139ab97d540195229f372-8.png', '4d637018815139ab97d540195229f372-11.png', '4d637018815139ab97d540195229f372-12.png'] 
bin1=['7bde818d2cbd21f3bac465483662a51d-0.png', '7bde818d2cbd21f3bac465483662a51d-3.png', '7bde818d2cbd21f3bac465483662a51d-10.png', '7bde818d2cbd21f3bac465483662a51d-12.png']
bin2=['8ab06d642437f3a77d8663c09e4f524d-0.png', '8ab06d642437f3a77d8663c09e4f524d-3.png', '8ab06d642437f3a77d8663c09e4f524d-5.png', '8ab06d642437f3a77d8663c09e4f524d-8.png', '8ab06d642437f3a77d8663c09e4f524d-9.png', '8ab06d642437f3a77d8663c09e4f524d-13.png']
display1=['2d5d4d79cd464298566636e42679cc7f-0.png', '2d5d4d79cd464298566636e42679cc7f-1.png', '2d5d4d79cd464298566636e42679cc7f-2.png', '2d5d4d79cd464298566636e42679cc7f-5.png', '2d5d4d79cd464298566636e42679cc7f-6.png', '2d5d4d79cd464298566636e42679cc7f-7.png', '2d5d4d79cd464298566636e42679cc7f-9.png', '2d5d4d79cd464298566636e42679cc7f-11.png', '2d5d4d79cd464298566636e42679cc7f-13.png']
display2=['17226b72d812ce47272b806070e7941c-1.png', '17226b72d812ce47272b806070e7941c-3.png', '17226b72d812ce47272b806070e7941c-4.png', '17226b72d812ce47272b806070e7941c-5.png', '17226b72d812ce47272b806070e7941c-6.png', '17226b72d812ce47272b806070e7941c-8.png', '17226b72d812ce47272b806070e7941c-9.png', '17226b72d812ce47272b806070e7941c-13.png']
printer1= ['7c1ac983a6bf981e8ff3763a6b02b3bb-0.png', '7c1ac983a6bf981e8ff3763a6b02b3bb-1.png', '7c1ac983a6bf981e8ff3763a6b02b3bb-4.png', '7c1ac983a6bf981e8ff3763a6b02b3bb-5.png', '7c1ac983a6bf981e8ff3763a6b02b3bb-8.png', '7c1ac983a6bf981e8ff3763a6b02b3bb-10.png', '7c1ac983a6bf981e8ff3763a6b02b3bb-13.png']
printer2= ['2135295ad1c580e7ffbff4948728b4f5-0.png', '2135295ad1c580e7ffbff4948728b4f5-1.png', '2135295ad1c580e7ffbff4948728b4f5-2.png', '2135295ad1c580e7ffbff4948728b4f5-3.png', '2135295ad1c580e7ffbff4948728b4f5-6.png', '2135295ad1c580e7ffbff4948728b4f5-7.png', '2135295ad1c580e7ffbff4948728b4f5-8.png', '2135295ad1c580e7ffbff4948728b4f5-9.png', '2135295ad1c580e7ffbff4948728b4f5-13.png']
bottle1= ['9eccbc942fc8c0011ee059e8e1a2ee9-6.png', '9eccbc942fc8c0011ee059e8e1a2ee9-7.png', '9eccbc942fc8c0011ee059e8e1a2ee9-10.png', '9eccbc942fc8c0011ee059e8e1a2ee9-11.png']
bottle2= ['62451f0ab130709ef7480cb1ee830fb9-0.png', '62451f0ab130709ef7480cb1ee830fb9-1.png', '62451f0ab130709ef7480cb1ee830fb9-6.png', '62451f0ab130709ef7480cb1ee830fb9-8.png']

#bottle3= ['d851cbc873de1c4d3b6eb309177a6753_0.png', 'd851cbc873de1c4d3b6eb309177a6753_6.png', 'd851cbc873de1c4d3b6eb309177a6753_11.png', 'd851cbc873de1c4d3b6eb309177a6753_12.png']
paper1=['3dw.b463019e5f3b9a9b5c6bfbdfe6a8f99.png', '3dw.b463019e5f3b9a9b5c6bfbdfe6a8f99-2.png', '3dw.b463019e5f3b9a9b5c6bfbdfe6a8f99-3.png', '3dw.b463019e5f3b9a9b5c6bfbdfe6a8f99-4.png']
paper2=['3dw.14b4294e99cb10215f606243e56be258.png', '3dw.14b4294e99cb10215f606243e56be258-2.png', '3dw.14b4294e99cb10215f606243e56be258-3.png', '3dw.14b4294e99cb10215f606243e56be258-4.png']
book1=['3dw.13d22e3b3657e229ce6cd687d82659e9.png', '3dw.13d22e3b3657e229ce6cd687d82659e9-2.png', '3dw.13d22e3b3657e229ce6cd687d82659e9-3.png', '3dw.13d22e3b3657e229ce6cd687d82659e9-4.png']
book2= ['3dw.1d493a57a21833f2d92c7cdc3939488b.png', '3dw.1d493a57a21833f2d92c7cdc3939488b-2.png', '3dw.1d493a57a21833f2d92c7cdc3939488b-3.png', '3dw.1d493a57a21833f2d92c7cdc3939488b-4.png']
table1=['f9f9d2fda27c310b266b42a2f1bdd7cf-4.png', 'f9f9d2fda27c310b266b42a2f1bdd7cf-10.png', 'f9f9d2fda27c310b266b42a2f1bdd7cf-11.png', 'f9f9d2fda27c310b266b42a2f1bdd7cf-13.png']
table2=['7807caccf26f7845e5cf802ea0702182-1.png', '7807caccf26f7845e5cf802ea0702182-6.png', '7807caccf26f7845e5cf802ea0702182-11.png', '7807caccf26f7845e5cf802ea0702182-12.png']
box1=['3dw.f0f1419ffe0e4475242df0a63deb633.png', '3dw.f0f1419ffe0e4475242df0a63deb633-2.png', '3dw.f0f1419ffe0e4475242df0a63deb633-3.png', '3dw.f0f1419ffe0e4475242df0a63deb633-4.png']
box2=['3dw.405e820d9717e1724cb8ef90d0735cb6.png', '3dw.405e820d9717e1724cb8ef90d0735cb6-2.png', '3dw.405e820d9717e1724cb8ef90d0735cb6-3.png', '3dw.405e820d9717e1724cb8ef90d0735cb6-4.png']
window1=['3dw.2f322060f3201f71caf432acbbd622b.png', '3dw.2f322060f3201f71caf432acbbd622b-2.png']
window2=['3dw.e884d2ee658acf6faa0e334660b67084.png', '3dw.e884d2ee658acf6faa0e334660b67084-2.png', '3dw.e884d2ee658acf6faa0e334660b67084-3.png', '3dw.e884d2ee658acf6faa0e334660b67084-4.png' ]
door1=['3dw.f40d5808bf78f8003f7c9f4b711809d.png', '3dw.f40d5808bf78f8003f7c9f4b711809d-2.png']
door2=['3dw.c67fa55ac55e0ca0a3ef625a8daeb343.png', '3dw.c67fa55ac55e0ca0a3ef625a8daeb343-2.png']
sofa1=['4820b629990b6a20860f0fe00407fa79-0.png', '4820b629990b6a20860f0fe00407fa79-7.png', '4820b629990b6a20860f0fe00407fa79-9.png', '4820b629990b6a20860f0fe00407fa79-13.png']
sofa2=['87f103e24f91af8d4343db7d677fae7b-0.png', '87f103e24f91af8d4343db7d677fae7b-6.png', '87f103e24f91af8d4343db7d677fae7b-7.png', '87f103e24f91af8d4343db7d677fae7b-12.png']
lamp1=['3dw.3c5db21345130f0290f1eb8f29abcea8.png', '3dw.3c5db21345130f0290f1eb8f29abcea8-2.png']
lamp2=['6770adca6c298f68fc3f90c1b551a0f7-4.png', '6770adca6c298f68fc3f90c1b551a0f7-6.png', '6770adca6c298f68fc3f90c1b551a0f7-8.png', '6770adca6c298f68fc3f90c1b551a0f7-12.png']



#Macro, i.e., same object class
chairs=  list(set().union(chair1,chair2))
plants=  plant1
bins=  list(set().union(bin1,bin2))
displays =  list(set().union(display1,display2))
printers = list(set().union(printer1, printer2))

bottles = list(set().union(bottle1, bottle2)) #bottle3))
papers = list(set().union(paper1, paper2)) #bottle3))
books = list(set().union(book1, book2)) #bottle3))
tables = list(set().union(table1, table2)) #bottle3))
boxes = list(set().union(box1, box2)) #bottle3))
windows = list(set().union(window1, window2)) #bottle3))
doors = list(set().union(door1, door2)) #bottle3))
sofas = list(set().union(sofa1, sofa2)) #bottle3))
lamps = list(set().union(lamp1, lamp2)) #bottle3))

#all_ids = list(set().union(chairs,plants,bins)) # displays, printers))
all_ids = list(set().union(chairs, bottles, papers, books, tables, boxes, windows, doors, sofas, lamps))
#############################################################################

object_list = [chairs, bottles, papers, books, tables, boxes, windows, doors, sofas, lamps]
flags =  ["chairs", 'bottles', 'papers', 'books', 'tables', 'boxes', 'windows', 'doors', 'sofas', 'lamps']

micro_object_list = [(chair1, chair2), (bottle1, bottle2), (paper1, paper2), (book1, book2), (table1, table2), (box1, box2), (window1, window2), (door1, door2), (sofa1, sofa2), (lamp1, lamp2)]

randomized = False

def mainContour(image, flag):

    '''
    Extract most prominent shape from given image
    '''
    if flag =='input':
        ret, thresh = cv2.threshold(image, 0, 255,0)
    elif flag=='reference':
        ret, thresh = cv2.threshold(image, 127, 255,1)

    _, contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    areas = [cv2.contourArea(cnt) for cnt in contours]
  
    try: 
        idx = areas.index(max(areas))    

    except Exception as e:

        return contours[0]

    return contours[idx]



def cropToC(image, contour):

    alpha=0.0 # transparent overlay
    mask = np.zeros_like(image) # Create mask where white is what we want, black otherwise
    
    #print(type(mask))
    #print(type(contour))
    cv2.drawContours(mask, [contour], 0, 255, -1) # Draw filled contour in mask
        

    mask = np.invert(mask)
    #out = np.zeros_like(image) # Extract out the object and place into output image
    #Invert Black  with White (ShapeNets backgrounds are white) 
    #out = np.invert(out)
    out = image.copy()
    #overlay = image.copy()
    #Add transarent mask on top
    cv2.addWeighted(mask, alpha, out, 1 - alpha,
		0, out)
    out[np.where((mask == [255,255,255]).all(axis = 2))] = [255,255,255]
    
    # Now crop
    x = np.where(mask == [0,0,0])[0]
    y = np.where(mask == [0,0,0])[1]
    try:
        (topx, topy) = (np.min(x), np.min(y))
        (bottomx, bottomy) = (np.max(x), np.max(y))

        out= out[topx:bottomx+1, topy:bottomy+1]


    except Exception as e:
        print(str(e))
        sys.exit(0)
    '''    
    cv2.imshow('', out)
    cv2.waitKey(8000)
    sys.exit(0)
    '''
    return out

###########################################
#Methods for feature description
###########################################

def SIFT(gray):

    #Returns kp = list of keypoints
    #des = numpy array of dimensions kp x 128    

    sift = cv2.SIFT()
    kp, des = sift.detectAndCompute(gray,None)
    
    return kp, des

def SURF():
    
    # Create SURF object. You can specify params here or later.
    # Here I set Hessian Threshold to 400
    surf = cv2.xfeatures2d.SURF_create(400)
    surf.extended = True  # making the size of the descriptor 128
    # Find keypoints and descriptors directly
    kp, des = surf.detectAndCompute(img,None)    

    return kp, des


def BRIEF():

    brief = cv2.BriefDescriptorExtractor_create()
    kp, des = brief.compute(img, kp)

    return kp, des


def ORB(gray):

    orb = cv2.ORB_create()
    kp, des = orb.detectAndCompute(gray,None)

    return kp, des

def FAST(gray):

    #TO-DO check nonmax suppression
    fast = cv2.FastFeatureDetector_create()
    kp, des = fast.detectAndCompute(gray,None)

    return kp, des

def baseline_method(list_allids):
    
    return random.choice(lis_allids)

def featureMatch(des1,des2, method, flag=0, homography= True, K=2, thresh=0.75):

    '''
    Brute-force (0), FLANN matching (1)
    with or without homography

    '''
    if flag == 0 and method =='SIFT':

        #Brute force for SIFT
        # BFMatcher with default params

        bf = cv2.BFMatcher()
        
        matches = bf.knnMatch(des1,des2, K)
  
        '''
        # Apply ratio test
        good = []
        for m,n in matches:

            if m.distance < thresh*n.distance:

                good.append([m])

        return good
        '''

    elif flag == 0  and method =='ORB':
    
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
        # Match descriptors.
        matches = bf.match(des1,des2)
    
        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)
     
    else:
        #FLANN 
        
        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary
  
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(des1,des2,K)

    #Probably not needed yet (i.e., find specific object in the image)
    #if homography:
        

    #Return list of matches between the 2 input descriptors
    return matches 

        
if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument("imgpath", help="Provide path to the images to be classified") 
    parser.add_argument("modelpath", help="Provide path to the reference models")
    #parser.add_argument("pathtomasks", help="Provide path to the segmented objects to be evaluated")
    parser.add_argument("outpath", help="Provide path to output the similarity scores")
    parser.add_argument("method", help="Choose between SIFT, SURF, BRIEF, ORB")
    parser.add_argument('--skipfolders', required=False, nargs='*', 
    help='Optional: name of folders to be skipped. For example to skip folders named images1 and images4 type "--skipfolders images1 images4" Please do not provide a full path but just the folder name')

    args =parser.parse_args()

    #CHANGED: Ended up implementing dataset pre-proc in Matlab directly
    #see maskfiles.m

    #Imgs are read from the BYUDepth data set here
    #images = loadNYU(args.imgpath).T    #We have to take the transpose here from what it returns 

    #print(images.shape)
    skip_folders=args.skipfolders    
    
    modelfiles = os.listdir(args.modelpath)
    modelpaths = [os.path.join(args.modelpath, mfile) for mfile in modelfiles]

    for folder in os.listdir(args.imgpath):


        if folder =='bins' or folder=='plants':
            continue

        if skip_folders is not None and folder in skip_folders:
            continue     

        start = time.time()
        accuracy = 0.0     
        correct =0

        objectn = os.listdir(os.path.join(args.imgpath, folder))
        objcat = folder
    
        objectpaths = [os.path.join(args.imgpath, folder, name) for name in objectn]

        #Recap all in a csv
        wrtr = csv.writer(open(os.path.join(args.outpath, 'SIFT_BF_results_recap_%s.csv' % objcat), 'w'))
        #Write header
        wrtr.writerow(["imageid", 'category', 'bestmatch', 'score', 'mean', 'median', 'stdev', 'max', 'predicted', 'correct?'])
    
        print("Here we go, currently evaluating: %s" % objcat)

        for filep in objectpaths:

            row=[]
            l = filep.split("/")
            fname = l[len(l)-1]

            #Read image in grayscale
            objimg = cv2.imread(filep, 0) 
            objrgb = cv2.imread(filep, 1) 

 
            glob_min =1000.0
            glob_max =0.0
        
            row.append(fname)
            row.append(objcat)
            
            avgs =[] 


            #Compare with all Shapenet models
            scores =[]    
    
            for modelp in modelpaths: 

            
                lm = modelp.split('/')             
                modname = lm[len(lm)-1]


                mimage = cv2.imread(modelp, 0) #reads in grayscale
                mrgb = cv2.imread(modelp, 1)
            

                if args.method == 'SIFT':

                    kp, des = SIFT(objimg)
                    kpm, desm = SIFT(mimg)                    
                    #Get list of matches that passed ratio test

                elif args.method == 'ORB':

                    kp, des = ORB(objimg)
                    kpm, desm = ORB(mimg)

                elif args.method == 'SURF':

                    kp, des = SURF(objimg)
                    kpm, desm = SURF(mimg)

                elif args.method == 'BRIEF':

                    kp, des = BRIEF(objimg)
                    kpm, desm = BRIEF(mimg)

                elif args.method == 'FAST':

                    kp, des = FAST(objimg)
                    kpm, desm = FAST(mimg)
     
                good_ones = featureMatch(des, desm, args.method)

     
                #TO_DO: redefine strategy to compute winner
                            

                score = alpha*shapescore + beta*clrscore

                if score < glob_min:
                    glob_min = score
                    obj_min = modname
         
                    
                scores.append(score)

                
                if randomized:

                    obj_min = baseline_method(all_ids)

                row.append(obj_min)
                row.append(glob_min)
        
        
                #Add stats on scores
                row.append(stat.mean(scores))
                row.append(stat.median(scores))
                row.append(stat.stdev(scores))
                row.append(max(scores))
                
                #Output predicted cat
        
                if obj_min in chairs:
                    pred='chairs'

                elif obj_min in bins:
                    pred='bins'
        
                elif obj_min in plants:
                    pred='plants'
        
                elif obj_min in bottles:
                    pred='bottles'

                elif obj_min in papers:
                    pred='papers'

                elif obj_min in books:
                    pred='books'

                elif obj_min in tables:
                    pred='tables'


                elif obj_min in boxes:
                    pred='boxes'


                elif obj_min in windows:
                    pred='windows'


                elif obj_min in doors:
                    pred='doors'


                elif obj_min in sofas:
                    pred='sofas'


                elif obj_min in lamps:
                    pred='lamps'


                '''

                if obj_max in chairs:
                    pred='chairs'

                elif obj_max in bins:
                    pred='bins'
        
                elif obj_max in plants:
                    pred='plants'
                '''
        

                row.append(pred)

                if pred==objcat:
 
                    row.append(1)
                    correct +=1 
                else:
                    row.append(0)

                wrtr.writerow(row) 

                 

        print(correct)
        print(len(objectn))
        accuracy = float(correct/ len(objectn)) 
        print("Tot accuracy is %f " % accuracy )

        print("Object class complete...took %f seconds" % float(time.time() - start))
