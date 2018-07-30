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
import logging

#1. Background subtraction
def backgr_sub(depth_image):
    
    '''
    capture = cv2.VideoCapture(depth_video)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    subtr = cv2.bgsegm.createBackgroundSubtractorGMG()

    while(1):
        ret, frame = capture.read()
        fgmask = subtr.apply(frame)
        #Applying morphological opening to the output 
        #for auxiliary noise removal
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        cv2.imshow('frame',fgmask)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    
    capture.release()
    cv2.destroyAllWindows()
    '''
    
    #Removing anything greater than max peak in depth histogram    
    #Excluding 0.0, i.e., no values
    #cm = plt.cm.get_cmap('gray')

    n, bins, patches = plt.hist(depth_image.ravel(), 'auto', [1,10], color='green', edgecolor='black', linewidth=1)

    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    cm = plt.cm.get_cmap('gray')
    
    # scale values to interval [0,1]
    col = bin_centers - min(bin_centers)
    col /= max(col)

    '''
    for c, p in zip(col, patches):

        plt.setp(p, 'facecolor', cm(c))

        #print(str(cm(c)[:3]))

    plt.xlim([1,10])
    plt.ylim([1,30000]) 
    plt.show()
    plt.close()
    plt.clf()
    '''
    print(n.max())
    max_ind = n.tolist().index(n.max())
    print(max_ind)
    print(n)
    print(bins)
    #sys.exit(0)
    #Edges of max bin
    left_e = bins[max_ind+1]
    right_e = bins[max_ind+2] 

    print(left_e)
    print(right_e)

    print(depth_image)
    #"White out" if above threshold, i.e., shift to 10 m
    
    depth_image[depth_image > right_e] = 10.0  
    
    
    print(depth_image)
    #sys.exit(0)
    #From depth values to coloring:
    thresholds = bins    
    
    for i, (c,p) in enumerate(zip(col, patches)):
        
        r, g, b = cm(c)[:3]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        #print(gray)
        #sys.exit(0)
        if i!=0:

            
            k= i-1 
            if thresholds[k]!=0.0:
                depth_image[np.logical_and(depth_image>= thresholds[k], depth_image <thresholds[i])] = gray

    denoised = small_blob_rem(depth_image)

    return depth_image, denoised



def small_blob_rem(imgd):

    #find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(imgd.astype(np.uint8), connectivity=8)
    #connectedComponentswithStats yields every seperated component with information on each of them, such as size
    #the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
    sizes = stats[1:, -1]; nb_components = nb_components - 1

    # minimum size of particles we want to keep (number of pixels)
    #here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
    min_size = 50  

    #your answer image
    #img2 = np.zeros((output.shape))
    img2= imgd.copy()
    #for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):

       if sizes[i] >= min_size:
           print('Remove')
           img2[output == i + 1] = 0.2989 * 255 + 0.5870 * 255 + 0.1140 * 255

    return img2




if __name__ == '__main__':

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
        #print(bag.get_message_count())
        #sys.exit(0)
        #topics = bag.get_type_and_topic_info()[1].keys()
        #print(topics)
        #info_dict = yaml.load(subprocess.Popen(['rosbag', 'info', '--yaml', path_to_bag], stdout=subprocess.PIPE).communicate()[0])
        #print(info_dict) 
        bridge = CvBridge()
        
        img_mat =[(bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough"), str(msg.header.stamp.nsecs)) for topic, msg, t in bag.read_messages()]

        #print(len(img_mat))

        #sys.exit(0)        
        #print(bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough").dtype)
        bag.close()
        
        logging.info("Rosbag imported, took %f seconds" % float(time.time() -start))   
        #print(img_mat[0])
        #sys.exit(0)

    else:  #or read from image dir

        files = os.listdir(args.imgpath)
        logging.info("Reading from image folder...")       

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
            
                logging.info("Problem while opening img %s" % f)
                print(str(e))
            
                #Skip corrupted or zero-byte images

                continue    

    #backgr_sub('./video-depth.avi')


    for k, (img,stamp) in enumerate(img_mat):

        print(stamp)
        print(img.dtype)
        #sys.exit(0)
        #cv2.imwrite('/mnt/c/Users/HP/Desktop/KMI/out-canny/depth-asis.png',img)
        #print(img.shape) 
        #print(img.dtype)

        #For depth images
        #Trying to convert back to mm from uint16
        height, width = img.shape        

        imgd = np.uint32(img)
        imgd = imgd*0.001
        #print(imgd[100,:])
        print(np.max(imgd))
        print(np.min(imgd))
        
        #print(imgd)
        
        imgd, denoised_i= backgr_sub(np.float32(imgd))

        #imgd = imgd.astype(np.uint8)        
        plt.imshow(imgd, cmap = plt.get_cmap('gray'))
        plt.show()
        plt.clf()
        

        plt.imshow(denoised_i, cmap = plt.get_cmap('gray'))
        plt.show()
        plt.clf()
        
        #sys.exit(0)        
        
        logging.info("Resolution of your input images is "+str(width)+"x"+str(height))        

    
    
    logging.info("Complete...took %f seconds" % float(time.time() - start))
