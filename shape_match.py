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

inverseNeeded = False
randomized = False
micro = False
macro = True

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

        #print(modelp)
    
        #lm = modelp.split('/')             
        #modname = lm[len(lm)-1]

        #comparison["similarities"] =[]   

        mimage = cv2.imread(modelp, 0)
        mrgb = cv2.imread(modelp, 1)
            
        shape2 = mainContour(mimage, 'reference')
            
        #Perform a pointwise comparison within each couple
        shapescore= shapeMatch(shape1, shape2) 
       
        #Crop images to contour 
        mrgb = cropToC(mrgb, shape2)
             
        
        #sys.exit(0)
        #print(filep)

        #WEIGHTS are INITIALIZED HERE!
        alpha = 1.0
        beta=  1.0

        clrscore, flag = featureMatch(objrgb, mrgb)
     
        if flag:

            #print("Inverting opposite trend scores!")
            clrscore = 1./clrscore   
            
        score = alpha*shapescore + beta*clrscore

        scores.append(score)

            
    return scores

def random_pick_from(pathlist, sizepick=1000):

    patharray = np.asarray(pathlist)

    return np.random.choice(patharray, size= sizepick).tolist()

        
if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument("imgpath", help="Provide path to the images to be classified") 
    parser.add_argument("modelpath", help="Provide path to the reference models")
    #parser.add_argument("pathtomasks", help="Provide path to the segmented objects to be evaluated")
    parser.add_argument("outpath", help="Provide path to output the similarity scores")
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
    
        #Downsample the chairs
        '''
        if objcat =="chairs":

            objectn = random_pick_from(objectn, 1000)
        '''
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
        objectpaths = [os.path.join(args.imgpath, folder, name) for name in objectn]

        #Recap all in a csv
        wrtr = csv.writer(open(os.path.join(args.outpath, 'shapevshape_macro_l3hell_results_recap_%s.csv' % objcat), 'w'))
        #Write header
        wrtr.writerow(["imageid", 'category', 'bestmatch', 'score', 'mean', 'median', 'stdev', 'max', 'predicted', 'correct?'])
    
        print("Here we go, currently evaluating: %s" % objcat)

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

                shape1 = mainContour(objimg, 'reference') #'input')

                #cv2.drawContours(objrgb, [shape1], 0, (0,255,0), 3)
                objrgb = cropToC(objrgb, shape1)

             
    
                #cv2.imshow('', objrgb)
                #cv2.waitKey(8000)
                #sys.exit(0)
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

                for type1, type2 in micro_object_list:
 
                    scores1 = microscore(type1, args.modelpath)
                    avg1 = sum(scores1)/len(scores1)
                    scores2 = microscore(type2, args.modelpath)
                    avg2 = sum(scores2)/len(scores2)
                    avgs.append(avg1)
                    avgs.append(avg2)

                glob_min = min(avgs)

                min_idx = avgs.index(min(avgs))

                #print(min_idx)
                
                if min_idx == 0 or min_idx == 1:

                    min_idx =0
                 
                elif min_idx % 2 == 0:
                    #Even number
                    min_idx = int(min_idx /2)
                    #print(min_idx) 

                elif min_idx % 2 != 0:
                    #Then it is od
                    
                    min_idx = int((min_idx -1)/2 )
                    #print(min_idx)

                    
                pred = flags[min_idx] 
                obj_min = flags[min_idx].split('s')[0]


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
        
                for slist in object_list:
            
                    scoretot = macroscore(slist, args.modelpath)
                    avgp = sum(scoretot)/len(scoretot)
                    avgs.append(avgp)

                glob_min = min(avgs)
                min_idx = avgs.index(min(avgs))

                pred = flags[min_idx]
                obj_min = pred.split('s')[0]

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
            
                    shape2 = mainContour(mimage, 'reference')
            
                    #Perform a pointwise comparison within each couple
                    shapescore= shapeMatch(shape1, shape2) 
                    
                    #Crop images to contour 
                    #cv2.drawContours(mrgb, [shape2], 0, (0,255,0), 3)
                    #print(modelp)
                    #print(type(mrgb))
                    #print(type(shape2))
                    mrgb = cropToC(mrgb, shape2)
             
    
                    #cv2.imshow('', mrgb)
                    #cv2.waitKey(8000)
                    #sys.exit(0)
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

        
                 
                '''
                with open(os.path.join(args.outpath, jname), 'w') as outf:

                    json.dump(simdict, outf, indent=4)

                #sys.exit(0)
                '''

        print(correct)
        print(len(objectn))
        accuracy = float(correct/ len(objectn)) 
        print("Tot accuracy is %f " % accuracy )

        print("Object class complete...took %f seconds" % float(time.time() - start))
