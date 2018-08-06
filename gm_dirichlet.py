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
from sklearn.mixture import BayesianGaussianMixture
import itertools
from scipy import linalg
import matplotlib as mpl
import random
import math

color_iter = itertools.cycle(plt.cm.rainbow(np.linspace(0,1,10)))

def chunks(l):
    
    for i in range(0, len(l)-1):

        yield l[i:i+2]

    
def plot_results(X, Y_, means, covariances, index, title):
    splot = plt.subplot()

    for i, (mean, covar, color) in enumerate(zip(means, covariances, color_iter)):

        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        
        print(u.shape)
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)
        
        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)
        
    plt.xticks(())
    plt.yticks(())
    plt.title(title)



def BGM_Dirichlet(imgd, outpath, stamp, sample_ratio =0.3):   

    #Image resize for sub-sampling
    #imgd = np.reshape(imgd,(imgd.shape[0]*imgd.shape[1], 1))    
    #print(imgd)

    n, bins, patches = plt.hist(imgd.ravel(), 'auto', [1,10], color='green', edgecolor='black', linewidth=1)
    cbins = chunks(bins)
    #Clean up histogram, ellipses will be plotted next
    plt.clf()

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
    #print(imgd)
    #sys.exit(0)
    #Reshape MxN depth matrix to have a list of pixels instead
    #imgd = np.reshape(imgd,(imgd.shape[0]*imgd.shape[1], 1))

    #print(imgd.shape)
    bgmm = BayesianGaussianMixture(n_components=10, covariance_type='full', max_iter=1000 ,n_init= 10, weight_concentration_prior_type='dirichlet_process').fit(imgd)

    plot_results(imgd, bgmm.predict(imgd), bgmm.means_, bgmm.covariances_, 0,
             'Gaussian Mixture')
    
    #And save resulting plot under given path
    plt.savefig(os.path.join(outpath, 'bgmm_%s.png' %stamp))
    plt.clf()
   
    


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
        #print(bag.get_message_count())
        #sys.exit(0)
        #topics = bag.get_type_and_topic_info()[1].keys()
        #print(topics)
        #info_dict = yaml.load(subprocess.Popen(['rosbag', 'info', '--yaml', path_to_bag], stdout=subprocess.PIPE).communicate()[0])
        #print(info_dict) 
        bridge = CvBridge()
        
        img_mat =[(bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough"), str(msg.header.stamp.secs)+str(msg.header.stamp.nsecs)) for topic, msg, t in bag.read_messages()]

        #print(len(img_mat))

        #sys.exit(0)        
        #print(bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough").dtype)
        bag.close()
        
        logging.info("Rosbag imported, took %f seconds" % float(time.time() -start))   
        #print(img_mat[0])
        #sys.exit(0)

    else:  #or read all bags in a dir
            
                    
        bag_paths = [os.path.join(args.imgpath, name) for name in os.listdir(args.imgpath)]

        for bpath in bag_paths:    

            try:

            except Exception as e:
            
                logging.info("Problem while opening img %s" % f)
                print(str(e))
            
                #Skip corrupted or zero-byte images

                continue    

    #backgr_sub('./video-depth.avi')


    for k, (img,stamp) in enumerate(img_mat):

        #if os.path.isfile(os.path.join('/mnt/c/Users/HP/Desktop/KMI/clean','%s.png') % stamp):
        #continue
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
        BGM_Dirichlet(imgd, args.outpath,stamp)

        sys.exit(0)
        
        logging.info("Resolution of your input images is "+str(width)+"x"+str(height))        

    
    
    logging.info("Complete...took %f seconds" % float(time.time() - start))
