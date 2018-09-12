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



#### A bunch of ugly hardcoded object IDs    ##############################

chairs=['9e14d77634cf619f174b6156db666192-0.png', '9e14d77634cf619f174b6156db666192-2.png', '9e14d77634cf619f174b6156db666192-5.png', '9e14d77634cf619f174b6156db666192-7.png', '9e14d77634cf619f174b6156db666192-10.png', '9e14d77634cf619f174b6156db666192-12.png', '49918114029ce6a63db5e7f805103dd-0.png', '49918114029ce6a63db5e7f805103dd-1.png' ,'49918114029ce6a63db5e7f805103dd-5 (1).png', '49918114029ce6a63db5e7f805103dd-5.png', '49918114029ce6a63db5e7f805103dd-6.png', '49918114029ce6a63db5e7f805103dd-9.png', '49918114029ce6a63db5e7f805103dd-11.png', '49918114029ce6a63db5e7f805103dd-13.png', 'c.png']
plants=['4d637018815139ab97d540195229f372-1.png', '4d637018815139ab97d540195229f372-3.png', '4d637018815139ab97d540195229f372-7.png', '4d637018815139ab97d540195229f372-8.png', '4d637018815139ab97d540195229f372-11.png', '4d637018815139ab97d540195229f372-12.png' ]
bins=['7bde818d2cbd21f3bac465483662a51d-0.png', '7bde818d2cbd21f3bac465483662a51d-3.png', '7bde818d2cbd21f3bac465483662a51d-10.png', '7bde818d2cbd21f3bac465483662a51d-12.png', '8ab06d642437f3a77d8663c09e4f524d-0.png', '8ab06d642437f3a77d8663c09e4f524d-3.png', '8ab06d642437f3a77d8663c09e4f524d-5.png', '8ab06d642437f3a77d8663c09e4f524d-8.png', '8ab06d642437f3a77d8663c09e4f524d-9.png', '8ab06d642437f3a77d8663c09e4f524d-13.png']

#############################################################################


inverseNeeded = False


def mainContour(image):

    '''
    Extract most prominent shape from given image
    '''
    ret, thresh = cv2.threshold(image, 0, 255,0)
    #ret, thresh = cv2.threshold(objimg, 127, 255,1)
    _, contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    areas = [cv2.contourArea(cnt) for cnt in contours]
   
    idx = areas.index(max(areas))    

    return contours[idx]



def shapeMatch(shape1, shape2):
    
    '''
    Performs comparison just based on shape of main contour,
    regardless of RGB or any other feature
    '''


    #model  = cv2.cvtColor(model,cv2.COLOR_GRAY2RGB)
    #cv2.drawContours(model, [shape2], 0, (0,255,0), 3)
    
    #cv2.imshow('', model)
    #cv2.waitKey(8000)
    #sys.exit(0)

    #Currently takes L3 method as reference
    return cv2.matchShapes(shape1, shape2,3,0.0)
    


def cropToC(image, contour):

    alpha=0.0 # transparent overlay
    mask = np.zeros_like(image) # Create mask where white is what we want, black otherwise
    
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
    
    #cv2.imshow('', out)
    #cv2.waitKey(0)

    return out

def featureMatch(inimg, refimg, flag=0):

    '''
    TODO: figure out if it should better have img
    cropped to contour instead

    '''

    
    #Histogram comparison as method
    if flag ==0:


        # extract a 3D RGB color histogram from the image,
	# using 8 bins per channel, normalize, and update
	# the index
	hist = cv2.calcHist([inimg], [0, 1, 2], None, [8, 8, 8],
		[0, 256, 0, 256, 0, 256])
	hist = cv2.normalize(hist, hist).flatten()

	hist2 = cv2.calcHist([refimg], [0, 1, 2], None, [8, 8, 8],
		[0, 256, 0, 256, 0, 256])
	hist2 = cv2.normalize(hist2, hist2).flatten()

        OPENCV_METHODS = (
	("Correlation", cv2.HISTCMP_CORREL), #the higher the better
	("Chi-Squared", cv2.HISTCMP_CHISQR), #the smaller the better
	("Intersection", cv2.HISTCMP_INTERSECT), #the higher the better
	("Hellinger", cv2.HISTCMP_BHATTACHARYYA)) #the smaller the better

        d = cv2.compareHist(hist2, hist, OPENCV_METHODS[0][1])

        if OPENCV_METHODS[0][1] == 0 or OPENCV_METHODS[0][1] == 2:
            
            inverseNeeded = True

    return d, inverseNeeded        

     
        
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

    #Recap all in a csv
    wrtr = csv.writer(open(os.path.join(args.outpath, 'l3corr_results_recap_chairs_0307.csv'), 'w'))
    #Write header
    wrtr.writerow(["imageid", 'category', 'bestmatch', 'score', 'mean', 'median', 'stdev', 'max', 'predicted', 'correct?'])
    
    for filep in objectpaths:

        row=[]
        simdict = {}
        l = filep.split("/")
        fname = l[len(l)-1]
        objcat = l[len(l)-2]

        #Read image in grayscale
        objimg = cv2.imread(filep, 0) 
    
        objrgb = cv2.imread(filep, 1) 
 
        simdict["img_id"] = fname
        simdict['obj_category']= objcat
        simdict['comparisons']=[]
        glob_min =1000.0
        glob_max =0.0
        
        row.append(fname)
        row.append(objcat)


        #Extract shape
        try:

            shape1 = mainContour(objimg)

            objrgb = cropToC(objrgb, shape1)
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
 
        scores =[]        

        #Compare with all Shapenet models
        for modelp in modelpaths: 

            comparison={} 
            
            lm = modelp.split('/')             
            modname = lm[len(lm)-1]

            comparison["compared_obj"] = modname
            #comparison["similarities"] =[]   

            mimage = cv2.imread(modelp, 0)
            mrgb = cv2.imread(modelp, 1)
            
            shape2 = mainContour(mimage)
            
            #Perform a pointwise comparison within each couple
            shapescore= shapeMatch(shape1, shape2) 
       
            #Crop images to contour 
            mrgb = cropToC(mrgb, shape2)
             
            #sys.exit(0)
            #print(filep)

            #WEIGHTS are INITIALIZED HERE!
            alpha =0.3
            beta= 0.7

            clrscore, flag = featureMatch(objrgb, mrgb)
     
            if flag:

                #print("Inverting opposite trend scores!")
                clrscore = 1./clrscore   
            
            score = alpha*shapescore + beta*clrscore

            #print(score)                   
            #sys.exit(0)

            comparison["similarity"]= score
            
            '''
            try:
                iterat, curr_min = min(comparison["similarities"])  

            
            except TypeError:

                #Not enough values yet
                curr_min = score
            '''
            
            if score < glob_min:
                glob_min = score
                obj_min = modname
            '''
            if score > glob_max:
                glob_max=score
                obj_max =modname
            '''
            scores.append(score)
            simdict['comparisons'].append(comparison)


        #Sort by descending similarity
        #simdict["comparisons"] = sorted(simdict["comparisons"],key=lambda x:(x[1],x[0]))
        
        
        #Add key field for most similar object 
        simdict["min"]=(obj_min, glob_min)

        row.append(obj_min)
        row.append(glob_min)
        '''
        row.append(obj_max)
        row.append(glob_max)
        '''
        #Output dictionary as JSON file
        jname = fname[:-3]+'json'
        
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
        else:
            row.append(0)

        wrtr.writerow(row) 
         
        '''
        with open(os.path.join(args.outpath, jname), 'w') as outf:

            json.dump(simdict, outf, indent=4)

        #sys.exit(0)
        '''



    logging.info("Complete...took %f seconds" % float(time.time() - start))
