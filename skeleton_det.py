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



    for k, (img,stamp) in enumerate(img_mat):

        print(stamp)
        print(img.dtype)
        #sys.exit(0)
        #cv2.imwrite('/mnt/c/Users/HP/Desktop/KMI/out-canny/depth-asis.png',img)
        #print(img.shape) 
        #print(img.dtype)


        #Trying to convert back to mm from uint16
        height, width = img.shape        

        imgd = np.uint32(img)
        imgd = imgd*0.001
        #print(imgd[100,:])
        print(np.max(imgd))
        print(np.min(imgd))
        
                
        logging.info("Resolution of your input images is "+str(width)+"x"+str(height))        

 
    logging.info("Complete...took %f seconds" % float(time.time() - start))
