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
import genpy

color_iter = itertools.cycle(plt.cm.rainbow(np.linspace(0,1,10)))

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


def get_histogram_plot(img_array, stampid, outpath):

    #Compute histogram of values 
    cm = plt.cm.get_cmap('plasma')
    
    # Plot histogram.
    #n, bins, patches = get_histogram(img_array, stampid, outpath)
    n, bins, patches = plt.hist(img_array.ravel(), 'auto', [0,10], color='green', edgecolor='black', linewidth=1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # scale values to interval [0,1]
    col = bin_centers - min(bin_centers)
    col /= max(col)
    
    for c, p in zip(col, patches):
        
        plt.setp(p, 'facecolor', cm(c)) 
    
    plt.xlim([0,10])
    plt.title('Histogram of Depth Values')
    plt.savefig(os.path.join(outpath,'hist_%s.png' % stampid))
    #Clean up histogram, for next plot types
    plt.clf()

    return n, bins, patches


def plot_results(X, Y_, means, covariances, index, title):
    splot = plt.subplot()

    for i, (mean, covar, color) in enumerate(zip(means, covariances, color_iter)):

        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        
        #print(u.shape)
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
    #plt.clf()



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
    
                                        
                #full_dlist = [(bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough"), str(msg.header.stamp.secs)+str(msg.header.stamp.nsecs)) for topic, msg, t in bag.read_messages(topics=['/camera/depth/image_raw'])]
                #img_mat.extend(full_dlist[0::2])    
                
                #rgb_mat.extend([(bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough"), str(msg.header.stamp.secs)+str(msg.header.stamp.nsecs)) for topic, msg, t in bag.read_messages() if topic=='/camera/rgb/image_raw'])
                
                bag.close()

            except Exception as e:
            
                logging.info("Problem while opening img %s" % bpath)
                print(str(e))
            
                #Skip corrupted or zero-byte images

                continue    

    #backgr_sub('./video-depth.avi')

    print(len(img_mat))
   
    print(len(rgb_mat))

    for k, (img,stamp) in enumerate(img_mat):

        print(stamp)
        #if os.path.isfile(os.path.join('/mnt/c/Users/HP/Desktop/KMI/clean','%s.png') % stamp):
        #continue
        orig_rgb, _ = rgb_mat[k]
        #print(orig_rgb.dtype)
        height, width = img.shape        

        imgd = np.uint32(img)
        imgd = imgd*0.001
        #print(imgd[100,:])
        #print(np.max(imgd))
        #print(np.min(imgd))
        get_histogram_plot(imgd, stamp, args.outpath)
        
        #print(imgd)
        BGM_Dirichlet(imgd, args.outpath,stamp, sample_ratio=0.1)

        #Read histogram and cluster plot from file
        histogram = cv2.imread(os.path.join(args.outpath, 'hist_%s.png' % stamp))
        clusterplt = cv2.imread(os.path.join(args.outpath, 'bgmm_%s.png' % stamp))
        
        sliced_image = rosimg_fordisplay(imgd, stamp, args.outpath)     

        plt.imshow(sliced_image, cmap = plt.get_cmap('plasma'))        
        plt.savefig(os.path.join(args.outpath, '_orig%s.png' % stamp))
        plt.clf()
        
   
        sliced_image = cv2.imread(os.path.join(args.outpath, '_orig%s.png' % stamp))
        #sliced_image =cv2.cvtColor(sliced_image, cv2.COLOR_BGR2GRAY)
        
        #Stack to previous two
        #histogram =cv2.cvtColor(histogram, cv2.COLOR_BGR2GRAY)
        #print(histogram.shape)

        upper = np.hstack((sliced_image, histogram))       
        lower = np.hstack((orig_rgb, clusterplt)) 
       
        full = np.vstack((upper, lower))
        cv2.imwrite(os.path.join(args.outpath,'final_%s.png' % stamp), full)
 
        logging.info("Resolution of your input images is "+str(width)+"x"+str(height))        

        #sys.exit(0)
    
    
    logging.info("Complete...took %f seconds" % float(time.time() - start))
