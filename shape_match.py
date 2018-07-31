import os
import time
import cv2
import argparse 
from PIL import Image
import numpy as np
import sys
from shapenet_select import findAndMatch
from pylab import arange, array, uint8 
from matplotlib import pyplot as plt
import rosbag
import subprocess, yaml
from cv_bridge import CvBridge, CvBridgeError
import json
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA



def preproc(imgm, img_wd, tstamp):
        

    img_new = imgm.copy() 
    img_new_l1 = imgm.copy()
    img_new_l2 = imgm.copy()
    img_new_l3 = imgm.copy()


    #print(imgm.dtype)
    #Obtaining depth values in mm
    img_new = np.uint32(img_new)
    img_new = img_new*0.001

    #print(np.may_share_memory(img_new, imgm))

    #get_histogram(imgm, tstamp) 
    #path = get_histogram(np.float32(img_new), tstamp) 
    #path = get_histogram(np.float32(img_new), tstamp) 
    #sys.exit(0)
    #plot =cv2.imread(path)
    #print(imgm.dtype)
    #plot =cv2.cvtColor(plot, cv2.COLOR_BGR2GRAY)
    #imgm =cv2.cvtColor(imgm, cv2.COLOR_BGR2GRAY)
 
    #print(plot.shape)
    #print(plot.dtype)
    #both = np.hstack((plot, imgm.astype(np.uint8)))   #.astype(np.uint8)))
    
    #new_im.save('test.jpg')    
    #substitute prior plot with both aligned 
    #cv2.imwrite(path, both)
    #sys.exit(0)
    '''
    twth= np.percentile(img_new, 25)
    fifth = np.percentile(img_new, 50)
    sevth = np.percentile(img_new, 75)

    #print(twth)   
    #print(fifth)
    #print(sevth)
 
    img_new_l1 = np.uint32(img_new_l1)
    img_new_l1 = img_new_l1*0.001
    #print(img_new_l1)
    img_new_l1[img_new_l1 > twth] = 255
    img_new_l1[img_new_l1 != 255] = 0
    #print(img_new_l1)
    
    #sys.exit(0)    


    img_new_l2 = np.uint32(img_new_l2)
    img_new_l2 = img_new_l2*0.001
    
    #img_new_l2[img_new_l2 == twth] = 255
    #img_new_l2[img_new_l2 != 255] = 0
    img_new_l2[img_new_l2 <= twth] = 255
    img_new_l2[img_new_l2 > fifth] = 255
    img_new_l2[img_new_l2 != 255] = 0
    
    img_new_l3 = np.uint32(img_new_l3)
    img_new_l3 = img_new_l3*0.001
    img_new_l3[img_new_l3 <= fifth] = 255
    img_new_l3[img_new_l3 > sevth] = 255
    img_new_l3[img_new_l3 != 255] = 0
    
    cv2.imwrite('/mnt/c/Users/HP/Desktop/KMI/out-canny/layers/depth-masked_l1%s.png' % tstamp,img_new_l1)
    cv2.imwrite('/mnt/c/Users/HP/Desktop/KMI/out-canny/layers/depth-masked_l2%s.png' % tstamp,img_new_l2)
    cv2.imwrite('/mnt/c/Users/HP/Desktop/KMI/out-canny/layers/depth-masked_l3%s.png' % tstamp,img_new_l3)
    #sys.exit(0)

    kernel = np.ones((10,10),np.uint8)
    #dilation = cv2.dilate(img_new,kernel,iterations = 10)
    op1 =  cv2.morphologyEx(img_new_l1, cv2.MORPH_OPEN, kernel)
    op2 =  cv2.morphologyEx(img_new_l2, cv2.MORPH_OPEN, kernel)
    op3 =  cv2.morphologyEx(img_new_l3, cv2.MORPH_OPEN, kernel)
    #closing =  cv2.morphologyEx(op, cv2.MORPH_CLOSE, kernel)

    aligned =np.hstack((img_new_l1,op1))
    cv2.imwrite('/mnt/c/Users/HP/Desktop/KMI/out-canny/masked/depth-masked_afterop_%s.png' % tstamp,aligned)
    aligned2 =np.hstack((img_new_l2,op2))
    cv2.imwrite('/mnt/c/Users/HP/Desktop/KMI/out-canny/masked/depth-masked_afterop_%s.png' % tstamp,aligned2)
    aligned3 =np.hstack((img_new_l3,op3))
    cv2.imwrite('/mnt/c/Users/HP/Desktop/KMI/out-canny/masked/depth-masked_afterop_%s.png' % tstamp,aligned3)
    
    edges=[]
    img_new = op1.astype(np.uint8)
    #jet = cv2.applyColorMap(equ,cv2.DIST_L2,5)
    jet = cv2.applyColorMap(img_new,cv2.DIST_L2,5)
    cv2.imwrite('/mnt/c/Users/HP/Desktop/KMI/out-canny/jet/jet_equ%s.png' % tstamp, jet)
    #sys.exit(0)
    #cv2.imwrite('/mnt/c/Users/HP/Desktop/KMI/out-canny/canny_'+f, canny) 
    #print(img_new.shape)

    __, contours1,hierarchy = cv2.findContours(img_new, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) 
    cv2.drawContours(jet, contours1, -1, (0, 0, 255), 3)

    cv2.imwrite('/mnt/c/Users/HP/Desktop/KMI/out-canny/cont/jet_cont%s.png' % tstamp, jet)
    
    img_new2 = op2.astype(np.uint8)
    #jet = cv2.applyColorMap(equ,cv2.DIST_L2,5)
    jet2 = cv2.applyColorMap(img_new2,cv2.DIST_L2,5)
    cv2.imwrite('/mnt/c/Users/HP/Desktop/KMI/out-canny/jet/jet_equ%s_2.png' % tstamp, jet2)
    #sys.exit(0)
    #cv2.imwrite('/mnt/c/Users/HP/Desktop/KMI/out-canny/canny_'+f, canny) 
    #print(img_new.shape)

    __, contours2,hierarchy = cv2.findContours(img_new2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) 
    cv2.drawContours(jet2, contours2, -1, (0, 0, 255), 3)

    cv2.imwrite('/mnt/c/Users/HP/Desktop/KMI/out-canny/cont/jet_cont%s_2.png' % tstamp, jet2)
    #edges=gray
    #print(cv2.connectedComponents(gray, 4, cv2.CV_32S))
    #cv2.imwrite('/mnt/c/Users/HP/Desktop/KMI/test',cv2.connectedComponents(gray, 4, cv2.CV_32S))
    
    img_new3 = op3.astype(np.uint8)
    #jet = cv2.applyColorMap(equ,cv2.DIST_L2,5)
    jet3 = cv2.applyColorMap(img_new3,cv2.DIST_L2,5)
    cv2.imwrite('/mnt/c/Users/HP/Desktop/KMI/out-canny/jet/jet_equ%s_3.png' % tstamp, jet3)
    #sys.exit(0)
    #cv2.imwrite('/mnt/c/Users/HP/Desktop/KMI/out-canny/canny_'+f, canny) 
    #print(img_new.shape)

    __, contours3, hierarchy = cv2.findContours(img_new3, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) 
    cv2.drawContours(jet3, contours3, -1, (0, 0, 255), 3)

    cv2.imwrite('/mnt/c/Users/HP/Desktop/KMI/out-canny/cont/jet_cont%s_3.png' % tstamp, jet3)
    
    return [contours1, contours2, contours3, img_new, img_new2, img_new3]
    '''
    return



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("imgpath", help="Provide path to your depth images/ or rosbag/ or npy file")
    parser.add_argument("outpath", help="Provide path to output imgs with drawn contours")
    args =parser.parse_args()

    start = time.time()


    files = os.listdir(args.imgpath)
    print("Reading from image folder...") 
      

    for f in files:
        
        if f[-3:] != 'png':
            #Skip config files, if any
            continue
        

        try:

            #Binarized image with only estimated foreground
            img_bin = cv2.imread(os.path.join(args.imgpath,f), cv2.IMREAD_UNCHANGED)
        
            img_bin =cv2.cvtColor(img_bin, cv2.COLOR_BGR2GRAY)
            __, contours,hierarchy = cv2.findContours(img_bin.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            cv2.drawContours(img_bin, contours, -1, (0, 0, 255), 3)

            cv2.imwrite('/mnt/c/Users/HP/Desktop/KMI/out-canny/cont/_cont/%s' % f, img_bin)

        except Exception as e:
            
            print("Problem while processing img %s" % f)
            print(str(e))
            
            #Skip corrupted or zero-byte images

            continue    


            '''
                cont1 = params[0] 
                cont2 = params[1] 
                cont3 = params[2] 
                img1 = params[3]
                img2 = params[4] 
                img3 = params[5]
 
                #Trying to match each layer separately with shapenet   
                simd1 = findAndMatch(contours2=cont1, img2=img1, fname=stamp+'_l1')

                with open(os.path.join('/mnt/c/Users/HP/Desktop/KMI/similarities', stamp+'_l1.json'), 'w') as js1:
                    json.dump(simd1, js1, indent=4)           

                simd2 = findAndMatch(contours2=cont2, img2=img2, fname=stamp+'_l2')
                with open(os.path.join('/mnt/c/Users/HP/Desktop/KMI/similarities', stamp+'_l2.json'), 'w') as js2:
                    json.dump(simd2, js2, indent=4)           
            
                simd3 = findAndMatch(contours2=cont3, img2=img3, fname=stamp+'_l3')

                with open(os.path.join('/mnt/c/Users/HP/Desktop/KMI/similarities', stamp+'_l3.json'), 'w') as js3:
                    json.dump(simd3, js3, indent=4)           
            

            '''

        
 
    print("Complete...took %f seconds" % float(time.time() - start))
