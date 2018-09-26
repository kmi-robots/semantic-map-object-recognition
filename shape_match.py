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


#Macro, i.e., same object class
chairs=  list(set().union(chair1,chair2))
plants=  plant1
bins=  list(set().union(bin1,bin2))
displays =  list(set().union(display1,display2))
printers = list(set().union(printer1, printer2))


all_ids = list(set().union(chairs,plants,bins)) # displays, printers))

#############################################################################


inverseNeeded = False
randomized = False
micro = False
macro = True

def mainContour(image):

    '''
    Extract most prominent shape from given image
    '''
    #ret, thresh = cv2.threshold(image, 0, 255,0)
    ret, thresh = cv2.threshold(image, 127, 255,1)
    _, contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    areas = [cv2.contourArea(cnt) for cnt in contours]
   
    idx = areas.index(max(areas))    

    '''
    cv2.drawContours(image, [contours[idx]], 0, (0,255,0), 3)
    
    cv2.imshow('', image)
    cv2.waitKey(8000)
    sys.exit(0)
    '''
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
    '''    
    cv2.imshow('', out)
    cv2.waitKey(8000)
    sys.exit(0)
    '''
    return out

def featureMatch(inimg, refimg, flag=0):

    '''
    TODO: figure out if it should better have img
    cropped to contour instead

    '''
    inverseNeeded= False
    
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

        d = cv2.compareHist(hist2, hist, OPENCV_METHODS[3][1])
        
        
        if OPENCV_METHODS[3][1] == 0 or OPENCV_METHODS[3][1] == 2:
            
            inverseNeeded = True

    return d, inverseNeeded        


def macroscore(namelist,path):

    #exactly the same, but on a broader list, 
    #just repeating for the sake of readability

    return microscore(namelist, path)

 
def microscore(namelist, path):

    scores=[]

    for modelname in namelist:
             
        modelp = os.path.join(path, modelname)

            
        lm = modelp.split('/')             
        modname = lm[len(lm)-1]

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
        alpha = 0.3
        beta=  0.7

        clrscore, flag = featureMatch(objrgb, mrgb)
     
        if flag:

            #print("Inverting opposite trend scores!")
            clrscore = 1./clrscore   
            
        score = alpha*shapescore + beta*clrscore

        scores.append(score)

            
    return scores

def random_pick_from(pathlist, sizepick=79):

    patharray = np.asarray(pathlist)

    return np.random.choice(patharray, size= sizepick).tolist()



        
if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument("imgpath", help="Provide path to the images to be classified") 
    parser.add_argument("modelpath", help="Provide path to the reference models")
    #parser.add_argument("pathtomasks", help="Provide path to the segmented objects to be evaluated")
    parser.add_argument("outpath", help="Provide path to output the similarity scores")
    args =parser.parse_args()

    start = time.time()
   
    accuracy = 0.0     
    correct =0
    #CHANGED: Ended up implementing dataset pre-proc in Matlab directly
    #see maskfiles.m

    #Imgs are read from the BYUDepth data set here
    #images = loadNYU(args.imgpath).T    #We have to take the transpose here from what it returns 

    #print(images.shape)
    
    modelfiles = os.listdir(args.modelpath)
    modelpaths = [os.path.join(args.modelpath, mfile) for mfile in modelfiles]

    objectn = os.listdir(args.imgpath)
    splits = args.imgpath.split('/')
    objcat = splits[len(splits)-1]
    
    #Part to comment/uncomment for downscaling#####
    '''
    if objcat =="chairs" or objcat =="plants":

        objectn = random_pick_from(objectn)

    ##############################################
    ''' 
    '''
    print(objectn)
    print(len(objectn))
    sys.exit(0)
    '''
    objectpaths = [os.path.join(args.imgpath, name) for name in objectn]

    #Recap all in a csv
    wrtr = csv.writer(open(os.path.join(args.outpath, 'shvsh_macro_l3hell_results_recap_bins_0307.csv'), 'w'))
    #Write header
    wrtr.writerow(["imageid", 'category', 'bestmatch', 'score', 'mean', 'median', 'stdev', 'max', 'predicted', 'correct?'])
    
    for filep in objectpaths:

        row=[]
        simdict = {}
        l = filep.split("/")
        fname = l[len(l)-1]
        #objcat = l[len(l)-2]

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
    
        avgs =[] 


        if micro:

            #Constrain the comparison by micro-category
            chair1s = microscore(chair1, args.modelpath)

            chair2s = microscore(chair2, args.modelpath)

            plant1s = microscore(plant1, args.modelpath)

            bin1s = microscore(bin1, args.modelpath)

            bin2s = microscore(bin2, args.modelpath)

            avgch1 = sum(chair1s)/len(chair1s)
            avgs.append(avgch1)
            avgch2 = sum(chair2s)/len(chair2s)
            avgs.append(avgch2)
            avgpl1 = sum(plant1s)/len(plant1s)
            avgs.append(avgpl1)
            avgb1 = sum(bin1s)/len(bin1s)
            avgs.append(avgb1)
            avgb2 = sum(bin2s)/len(bin2s)
            avgs.append(avgb2)
        
            glob_min = min(avgs)
            min_idx = avgs.index(min(avgs))

            if min_idx ==0 or min_idx==1:
                obj_min ='chair'
                pred='chairs'
        
            elif min_idx ==2:
                obj_min ='plant'
                pred='plants'

            elif min_idx ==3 or min_idx==4:
                obj_min ='bin'
                pred='bins'

        
            row.append(obj_min)
            row.append(glob_min)

            #Add stats on scores
            row.append(stat.mean(avgs))
            row.append(stat.median(avgs))
            row.append(stat.stdev(avgs))
            row.append(max(avgs))

            row.append(pred)

            if pred==objcat:
 
                row.append(1)
                correct +=1 
            else:
                row.append(0)

            wrtr.writerow(row) 


        #Constrain the comparison by macro-category
        elif macro:
        
            chairstot = macroscore(chairs, args.modelpath)

            plantstot = macroscore(plants, args.modelpath)

            binstot = macroscore(bins, args.modelpath)

            avgch = sum(chairstot)/len(chairstot)
            avgs.append(avgch)
            avgpl = sum(plantstot)/len(plantstot)
            avgs.append(avgpl)
            avgbin = sum(binstot)/len(binstot)
            avgs.append(avgbin)
        
            glob_min = min(avgs)
            min_idx = avgs.index(min(avgs))

            if min_idx ==0:
                obj_min ='chair'
                pred='chairs'
        
            elif min_idx ==1:
                obj_min ='plant'
                pred='plants'

            elif min_idx ==2:
                obj_min ='bin'
                pred='bins'

        
            row.append(obj_min)
            row.append(glob_min)

            #Add stats on scores
            row.append(stat.mean(avgs))
            row.append(stat.median(avgs))
            row.append(stat.stdev(avgs))
            row.append(max(avgs))

            row.append(pred)

            if pred==objcat:
 
                row.append(1)
                correct +=1 
            else:
                row.append(0)

            wrtr.writerow(row) 

        else:

                #Compare with all Shapenet models
                scores =[]    
    
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
                    
                    #score = clrscore #shapescore
                    score = alpha*shapescore + beta*clrscore

                    #print(score)                   
                    #sys.exit(0)

                    comparison["similarity"]= score
            
            
            
                    #try:
                    #    iterat, curr_min = min(comparison["similarities"])  

            
                    #except TypeError:

                        #Not enough values yet
                    #    curr_min = score
            
            
                    if score < glob_min:
                        glob_min = score
                        obj_min = modname
         
                    #elif randomized:
                    #    obj_min = random.choice(all_ids)
           
            
           
                    #if score > glob_max:
                    #    glob_max=score
                    #    obj_max =modname
            
            
                    scores.append(score)
                    simdict['comparisons'].append(comparison)

                '''

                #Sort by descending similarity
                #simdict["comparisons"] = sorted(simdict["comparisons"],key=lambda x:(x[1],x[0]))
        
                '''        
                if randomized:

                    obj_min = random.choice(all_ids)
                
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
                    correct +=1 
                else:
                    row.append(0)

                wrtr.writerow(row) 

        
                 
                '''
                with open(os.path.join(args.outpath, jname), 'w') as outf:

                    json.dump(simdict, outf, indent=4)

                #sys.exit(0)
                '''
    print(correct)
    print(len(objectn))
    accuracy = float(correct/ len(objectn)) 
    print("Tot accuracy is %f " % accuracy )

    logging.info("Complete...took %f seconds" % float(time.time() - start))
