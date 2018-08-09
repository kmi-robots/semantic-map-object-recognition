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
import PIL
from PIL import Image
import logging
from sklearn.mixture import BayesianGaussianMixture
import itertools
from scipy import linalg
import matplotlib as mpl
import random
import math
import genpy
import lightnet


color_iter = itertools.cycle(plt.cm.rainbow(np.linspace(0,1,10)))
model = lightnet.load('yolo')



def chunks(l):
    
    for i in range(0, len(l)-1):

        yield l[i:i+2]


def get_histogram(img_array, stampid, outpath):

    #Compute histogram of values 
    cm = plt.cm.get_cmap('plasma')
    #print(cm)
    # Plot histogram.
    n, bins, patches = plt.hist(img_array.ravel(), 'auto', [0,10], color='green', edgecolor='black', linewidth=1)

    plt.clf()
    return n, bins, patches


def BGM_Dirichlet(imgd, outpath, stamp, sample_ratio =0.3):   

    #Image resize for sub-sampling
    #imgd = np.reshape(imgd,(imgd.shape[0]*imgd.shape[1], 1))    
    #print(imgd)

    #n, bins, patches = plt.hist(imgd.ravel(), 'auto', [1,10], color='green', edgecolor='black', linewidth=1)
    
    n, bins, _ = get_histogram(imgd, stamp, outpath)

    cbins = chunks(bins)
    
    by_value =[]
    pixel_count=0

    for k, (left, right) in enumerate(cbins):
        
        values=[]

        #If non-empty bin
        if n[k] !=0.0:
            
            
            #Xs= np.where(np.logical_and(imgd>left, imgd<right))[0]
            
            #Ys= np.where(np.logical_and(imgd>left, imgd<right))[1]
           
            pixel_2 = np.where(np.logical_and(imgd>left, imgd<right))[0]
            #pixel_2 = np.fromiter((math.sqrt(x**2+y**2) for x,y in np.column_stack((Xs, Ys))), float)       
            #print(pixel_2) 
            
            values =np.column_stack((imgd[np.logical_and(imgd>left, imgd<right)], pixel_2)).tolist()
                         

            #And randomly take only 30% of those
            k= int(len(values)*sample_ratio)

            values = random.sample(values, k)
            
            by_value.extend(values)
            
         
        pixel_count+=len(values)

    pixel_zh = np.where(imgd==0.0)[0]
    #ZXs= np.where(imgd==0.0)[0]
    #ZYs= np.where(imgd==0.0)[1]
    #pixel_zh = np.fromiter((math.sqrt(x**2+y**2) for x,y in np.column_stack((ZXs, ZYs))), float)  
    
    zeros = np.column_stack((imgd[imgd == 0.0], pixel_zh)).tolist()
    
    h = int(len(zeros)*sample_ratio)
    zeros = random.sample(zeros, h)

    by_value.extend(zeros)
    
    imgd = np.asarray(by_value).reshape(-1, 2)
    
    bgmm = BayesianGaussianMixture(n_components=10, covariance_type='full', max_iter=1000 ,n_init= 10, weight_concentration_prior_type='dirichlet_process').fit(imgd)

    plot_results(imgd, bgmm.predict(imgd), bgmm.means_, bgmm.covariances_, 0,
             'Gaussian Mixture')
    
    #And save resulting plot under given path
    plt.savefig(os.path.join(outpath, 'bgmm_%s.png' %stamp))
    
   
def rosimg_fordisplay(img_d, stampid, outpath):

    #Define same colormap as the histograms 
    cm = plt.cm.get_cmap('plasma')

    n, bins, patches = get_histogram(imgd, stampid, outpath)
    #Cutoff values will be based on bin edges
    thresholds = bins
    
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    # scale values to interval [0,1]
    col = bin_centers - min(bin_centers)
    col /= max(col)
    
    #print(img_d)
    for i, (c,p) in enumerate(zip(col, patches)):
         
        r, g, b = cm(c)[:3]
        #gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        color = r* 65536 + g*256 + b
        if i!=0:
            
            n= i-1
 
            #if thresholds[n]!=0.0:
    
            img_d[np.logical_and(img_d>= thresholds[n], img_d <thresholds[i])] = color #gray
    
    return img_d    

def PILresize(imgpath, basewidth=416):

    img = Image.open(imgpath)
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    
    return img.resize((basewidth, hsize), PIL.Image.ANTIALIAS)
    


def find_bboxes(rgb_image, thr=0.1):

    #Object detection threshold defaults to 0.1 here
    img = lightnet.Image(rgb_image.astype(np.float32)) 
    boxes = model(img, thresh= thr)

    print(boxes)
    #Coordinates in YOLO are relative to center coordinates
    boxs_coord = [(int(x),int(y),int(w),int(h)) for cat,name, conf, (x,y,w,h) in boxes]  

    print(boxs_coord)
    return boxs_coord


def convert_boxes(box_list, shape, resolution):

    ow, oh, _ = shape
    tgt_w, tgt_h = resolution

    new_bx = []

    for x,y,w,h in box_list:

        print("Original: (%s, %s, %s, %s)" % (x,y,w,h))
        
        #Make them absolute from relative 
        x_ = x*tgt_w 
        y_ = y*tgt_h
        w_ = w*tgt_w
        h_ = h*tgt_h
        
        #print("Scaled: (%s, %s, %s, %s)" % (x_,y_,w_,h_))
        #And change coord system for later cropping
        x1 = (x_ - w_/2)/ow
        y1 = (y_ - h_/2)/oh
        x2 = (x_ + w_/2)/ow
        y2 = (y_ + h_/2)/oh

        #Add check taken from draw_detections method in Darknet's image.c
        if x1 < 0:
            x1= 0
        if x2 > ow-1:
            x2 = ow -1

        if y1 < 0:
            y1 = 0

        if y2 > oh -1: 

            y2 = oh -1 

        print("For ROI: (%s, %s, %s, %s)" % (x1,y1,x2,y2))
        new_bx.append((x1,y1,x2,y2))

    return new_bx 


def crop_img(rgb_image, boxs_coord, depthi):

    #Crop also the depth img for consistency
    return [(rgb_image[y:y+h, x:x+w], depthi[y:y+h, x:x+w]) for (x,y,w,h) in boxs_coord]




if  __name__ == '__main__':

    logging.getLogger().setLevel(logging.INFO)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("imgpath", help="Provide path to your depth images/ or rosbag/ or npy file")
    parser.add_argument("outpath", help="Provide path to output imgs with drawn contours")
    args =parser.parse_args()

    start = time.time()
    img_mat =[]

    #Read from pre-outputted npy file
    print(args.imgpath[-3:])
    
    if args.imgpath[-3:] == 'npy':
 
        #np.save("./rosdepth_with_stamps.npy", img_mat)
        img_mat = np.load(args.imgpath)
        #sys.exit(0)

    #Read from rosbag 
    elif args.imgpath[-3:] == 'bag':

        logging.info("Loading rosbag...")
        
        bag= rosbag.Bag(args.imgpath)
        bridge = CvBridge()
        
        img_mat =[(bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough"), str(msg.header.stamp.secs)+str(msg.header.stamp.nsecs)) for topic, msg, t in bag.read_messages()]

        bag.close()
        
        logging.info("Rosbag imported, took %f seconds" % float(time.time() -start))   

    else:  #or read all bags in a dir
            
        
        bag_paths = [os.path.join(args.imgpath, name) for name in os.listdir(args.imgpath)]
        img_mat=[]
        rgb_mat=[]

        for bpath in bag_paths:    

            if bpath[-8:] == 'orig.bag':
                continue
            try:
                
                bag= rosbag.Bag(bpath)
                bridge = CvBridge()

                copy_iter =  list(bag.read_messages(topics=['/camera/rgb/image_raw', '/camera/depth/image_raw']))
                #full_dlist = [(bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough"), str(msg.header.stamp.secs)+str(msg.header.stamp.nsecs), t) for topic, msg, t in bag.read_messages(topics=['/camera/depth/image_raw'])]
                for idx, (topic, msg, t) in enumerate(list(bag.read_messages(topics=['/camera/rgb/image_raw', '/camera/depth/image_raw']))):
                
                                        
                    if topic =='/camera/rgb/image_raw':
                        #RGB frame found
                        start_rgb = t 
                        rgb_msg = msg 
                        curr_i = idx
                        rgb_mat.extend([(bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough"), str(msg.header.stamp.secs)+str(msg.header.stamp.nsecs))])                         

                        try:
                            lookup = copy_iter[curr_i:curr_i+5]
                        except:
                            #end of list reached
                            lookup = copy_iter[curr_i:len(copy_iter)-1]

                        
                        for topic2, msg2, t2 in lookup:
                            
                            
                            if topic2 == '/camera/depth/image_raw':
                                #Closest next depth frame found
                                img_mat.extend([(bridge.imgmsg_to_cv2(msg2, desired_encoding="passthrough"), str(msg2.header.stamp.secs)+str(msg2.header.stamp.nsecs))])                         
                                break                                 
    
                                        
                
                bag.close()

            except Exception as e:
            
                logging.info("Problem while opening img %s" % bpath)
                print(str(e))
            
                #Skip corrupted or zero-byte images

                continue    


    #Consistency check
    if abs(len(img_mat) - len(rgb_mat)) > 1:

        logging.info('Input sets do not match!')
        sys.exit(0)
   

    for k, (img,stamp) in enumerate(img_mat):

        rgb_img, rstamp = rgb_mat[k]
       
        tgt_res = (416, 312)
        
        out = os.path.join(args.outpath, '%s.png' % stamp)
        cv2.imwrite(out, rgb_img)
 
        #Resize image to 416 x related height to work with YOLO  
        #Not efficient (writes to disk) but possibly more robust to different resolutions
        #rgb_res = PILresize(out)     
        #rgb_res.save(out)
        #rgb_res = cv2.imread(out)
        rgb_res= rgb_img.copy() 
        #Resize image to 416x312 to work with YOLO bbox detection 
        rgb_res = cv2.resize(rgb_res, dsize=tgt_res, interpolation=cv2.INTER_CUBIC)

        #Using Lightnet to extract bounding boxes 
        bboxes = find_bboxes(rgb_res)

        #Convert boxes back to original image resolution
        #And from YOLO format (center coord) to ROI format (top-left/bottom-right)
        n_bboxes = convert_boxes(bboxes, rgb_img.shape, tgt_res)        


        #Divide image up based on bboxes coordinates 
        obj_list = crop_img(rgb_img, n_bboxes, img.astype(np.float32))
        #Repeat the same on depth matrix too

        #Do pre-processing steps within each bounding box        
        for obj, dimg in obj_list:
            
            cv2.imwrite('/mnt/c/Users/HP/Desktop/test_crop.png', obj)

            cv2.imwrite('/mnt/c/Users/HP/Desktop/depth_crop.png', dimg)
 
        sys.exit(0)


    logging.info("Complete...took %f seconds" % float(time.time() - start))
