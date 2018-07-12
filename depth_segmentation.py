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

def preproc(imgm, img_wd, tstamp):
        

    #cv2.imwrite('/mnt/c/Users/HP/Desktop/KMI/out-canny/depth-asis.png',imgm)

    img_new = imgm.copy() 
    img_new_l1 = imgm.copy()
    img_new_l2 = imgm.copy()
    img_new_l3 = imgm.copy()

    img_new = np.uint32(img_new)
    img_new = img_new*0.001

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

def contour_det(gray_img):

    #Binarize 
    #ret,thresh = cv2.threshold(gray_img,127,255,cv2.THRESH_BINARY)    #+cv2.THRESH_OTSU)    
        
    kernel = np.ones((5,5),np.uint8)
   
    '''    
    dilation = cv2.dilate(thresh,kernel,iterations = 10)
    thresh=dilation 
    '''
    
    #closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    closing =  cv2.morphologyEx(gray_img, cv2.MORPH_CLOSE, kernel)
    #op = cv2.morphologyEx(gray_img, cv2.MORPH_OPEN, kernel)
    thresh=closing

    #Find contours 
    #retrieves all of the contours and reconstructs a full hierarchy of nested contours, no approx
    __, contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) 
    
    cv2.drawContours(thresh, contours, -1, (0, 0, 255), 3)
    #print(contours)
    return contours
    

def canny_edge(gray_img):

    return cv2.Canny(gray_img, 100, 105)


def draw_cnt(contours, img, fname, outpath):

    '''
    conts_new =[]

    
    for cnt in contours:
        
        #Approximate polygons         
        epsilon = 0.1*cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt,epsilon,True)

        cv2.drawContours(img, [approx], -1, (0, 0, 255), 3)


        conts_new.append(approx)    
    '''
    conts_new = contours
    cv2.drawContours(img, conts_new, -1, (0, 0, 255), 3)
    
    cv2.imwrite(os.path.join(outpath, fname), img)
    

    return conts_new

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("imgpath", help="Provide path to your depth images/ or rosbag")
    parser.add_argument("outpath", help="Provide path to output imgs with drawn contours")
    args =parser.parse_args()

    start = time.time()

    img_mat =[]

    #Read from rosbag 
    if args.imgpath[-3:] == 'bag':

        print("Loading rosbag...")
        
        bag= rosbag.Bag(args.imgpath)

        #topics = bag.get_type_and_topic_info()[1].keys()
        #print(topics)
        #info_dict = yaml.load(subprocess.Popen(['rosbag', 'info', '--yaml', path_to_bag], stdout=subprocess.PIPE).communicate()[0])
        #print(info_dict) 
        bridge = CvBridge()
        img_mat =[(bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough"), str(msg.header.stamp.secs)) for topic, msg, t in bag.read_messages()]
        bag.close()
        print("Rosbag imported, took %f seconds" % float(time.time() -start))   
        #print(img_mat[0])
        #sys.exit(0)

    else:  #or read from image dir

        files = os.listdir(args.imgpath)
        print("Reading from image folder...") 
      

        for f in files:
        
            if f[-3:] == 'ini':
                #Skip config files, if any
                continue
        

            try:
            #img_array1 = Image.open(os.path.join(args.imgpath,f))   #.convert('L')
            #img_np = np.asarray(img_array1)

                 img_array = cv2.imread(os.path.join(args.imgpath,f), cv2.IMREAD_UNCHANGED)
        
                 img_mat.append((img_array,f))

            except Exception as e:
            
                print("Problem while opening img %s" % f)
                print(str(e))
            
                #Skip corrupted or zero-byte images
                continue

    for img,stamp in img_mat:

        #print(img.shape) 
        #Trying to convert back to mm from uint16
        height, width = img.shape        

        imgd = np.uint32(img)
        imgd = imgd*0.001
        #print(imgd[100,:])
        print(np.max(imgd))
        print(np.min(imgd))
        
        print("Resolution of your input images is "+str(width)+"x"+str(height))        

        #print(img[100])        
        
        '''
        img =np.uint32(img)
        img = img*0.001

        print(np.max(img))
        print(np.min(img))
        sys.exit(0)   
        '''

        try:    

            #img = cv2.imdecode(img_np, cv2.IMREAD_UNCHANGED)
            params = preproc(img, imgd, stamp)
            cont1 = params[0] 
            cont2 = params[1] 
            cont3 = params[2] 
            img1 = params[3]
            img2 = params[4] 
            img3 = params[5]
 
            #Trying to match each layer separately with shapenet   
            findAndMatch(contours2=cont1, img2=img1, fname=stamp+'_l1')
            findAndMatch(contours2=cont2, img2=img2, fname=stamp+'_l2')
            findAndMatch(contours2=cont3, img2=img3, fname=stamp+'_l3')

            sys.exit(0)
            
            '''
            kernel = np.ones((5,5),np.uint8)
            #Segment
            contours = contour_det(edged)
            
            poly_contours = draw_cnt(contours, img_array, f, args.outpath)
            #cv2.imwrite('/mnt/c/Users/HP/Desktop/test.png', gray)
            #cv2.imwrite('/mnt/c/Users/HP/Desktop/test_ng.png', img_array)
            '''
            #sys.exit(0) 

        except Exception as e:

            print("Problem while processing ") 
            print(str(e))
            sys.exit(0)

        #Display, with segmentation

          

    #Dump to compressed output for future re-use
    print(len(img_mat))    
    #print(img_mat[0].shape)

    np.save("./depth_collected.npy", img_mat)
 
    print("Complete...took %f seconds" % float(time.time() - start))
