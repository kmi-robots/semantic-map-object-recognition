import os
import time
import cv2
import argparse 
from PIL import Image
import numpy as np
import sys
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
from skimage.future import graph
from skimage import color, filters, segmentation


color_iter = itertools.cycle(plt.cm.rainbow(np.linspace(0,1,10)))
model = lightnet.load('yolo')

def chunks(l):
    
    for i in range(0, len(l)-1):

        yield l[i:i+2]


def BGM_Dirichlet(imgd, rgb_image, sample_ratio =1.0):   

    n, bins, patches = plt.hist(imgd.ravel(), 'auto', [1,10], color='green', edgecolor='black', linewidth=1)

    cbins = chunks(bins)
    
    by_value =[]
    pixel_count=0
    by_mvalue= []

    for k, (left, right) in enumerate(cbins):
        
        values=[]
        mvalues=[]

        #If non-empty bin
        if n[k] !=0.0:
            
            pixel_2 = np.where(np.logical_and(imgd>left, imgd<right))[0]
            pixel_w = np.where(np.logical_and(imgd>left, imgd<right))[1]
           
            values = np.column_stack((imgd[np.logical_and(imgd>left, imgd<right)], pixel_2)).tolist()
            #Keeps track of widths also
            mvalues = np.column_stack((imgd[np.logical_and(imgd>left, imgd<right)], pixel_2, pixel_w)).tolist()
             
            '''
            #And randomly take only 30% of those
            k= int(len(values)*sample_ratio)

            values = random.sample(values, k)
            '''
            by_value.extend(values)
            by_mvalue.extend(mvalues)
         
        pixel_count+=len(values)


    pixel_zh = np.where(imgd==0.0)[0]
    pixel_zw = np.where(imgd==0.0)[1]
    
    zeros = np.column_stack((imgd[imgd == 0.0], pixel_zh)).tolist()
    
    #Keeps track of widths also
    mzeros = np.column_stack((imgd[imgd == 0.0], pixel_zh, pixel_zw)).tolist()
    
    #h = int(len(zeros)*sample_ratio)
    #zeros = random.sample(zeros, h)

    by_value.extend(zeros)
    by_mvalue.extend(mzeros)    

    #print(by_value)
    #print(by_value.shape)

    imgd = np.asarray(by_value).reshape(-1, 2)
    im_idx = np.asarray(by_mvalue).reshape(-1, 3)
    
    #print(imgd.shape)    
    #print(im_idx.shape)

    try:
        bgmm = BayesianGaussianMixture(n_components=3, covariance_type='full', max_iter=1000 ,n_init= 10, weight_concentration_prior_type='dirichlet_process').fit(imgd)

    except ValueError:
        #Not enough pixels in the box
        return
    #print(rgb_image.shape)

    #cm = plt.cm.get_cmap('jet')
    clusterm = np.zeros(rgb_image.shape, rgb_image.dtype)

    labels = np.zeros(rgb_image.shape[:2], rgb_image.dtype)
    print(labels.shape)
    #print(str(set(bgmm.predict(imgd))))
    #print(len(bgmm.predict(imgd)))

    for index, cluster_no in enumerate(bgmm.predict(imgd)):
        
        #Retrieve x and y in the original matrix 
        depth, height, width = by_mvalue[index]

        #print(rgb_image[0,1])        
        #Color based on cluster 
        r,g,b = plt.cm.get_cmap('Set3')(cluster_no, bytes=True)[:3]#.to_rgba(bytes=True)[:3]
      
        #print(r,g,b)
        #print(cm(cluster_no)[:3])
       
        #print(clusterm[int(height), int(width)])
        #print(r)

        clusterm[int(height), int(width)]= np.array([r,g,b], dtype=np.uint8) #[r,g,b] #cm(cluster_no)[:3] #(r,g,b)
        labels[int(height), int(width)]= cluster_no
 
        #print(clusterm[int(height), int(width)])
       
        #clusterm[int(height), int(width)][1]=g
        #clusterm[int(height), int(width)][2]=b 
        #On the original RGB matrix
        
        #print(clusterm[int(height), int(width)])

        #rgb_image[int(height)][int(width)] = np.array([r,g,b], dtype=np.uint8)

    #cmask = cv2.bitwise_and(redImg.astype(np.uint8), redImg, mask=cv2.cvtColor(imgd, cv2.COLOR_BGR2GRAY))
    #new_i = cv2.addWeighted(redMask, 1, rgb_img, 1, 0, rgb_img)
    
    return clusterm, labels

def BGM_Dir_depthonly(imgd, rgb_image, sample_ratio =1.0):   

    n, bins, patches = plt.hist(imgd.ravel(), 'auto', [1,10], color='green', edgecolor='black', linewidth=1)

    cbins = chunks(bins)
    
    by_value =[]
    pixel_count=0
    by_mvalue= []

    for k, (left, right) in enumerate(cbins):
        
        values=[]
        mvalues=[]
        
        #If non-empty bin
        if n[k] !=0.0:
            
            pixel_2 = np.where(np.logical_and(imgd>left, imgd<right))[0]
            pixel_w = np.where(np.logical_and(imgd>left, imgd<right))[1]
           
            values = imgd[np.logical_and(imgd>left, imgd<right)].tolist()
            #Keeps track of widths also
            mvalues = np.column_stack((imgd[np.logical_and(imgd>left, imgd<right)], pixel_2, pixel_w)).tolist()
             
            
            by_value.extend(values)
            by_mvalue.extend(mvalues)
         
        pixel_count+=len(values)

    print(len(by_value))

    pixel_zh = np.where(imgd==0.0)[0]
    pixel_zw = np.where(imgd==0.0)[1]
    
    zeros = imgd[imgd == 0.0].tolist()
    
    #Keeps track of widths also
    mzeros = np.column_stack((imgd[imgd == 0.0], pixel_zh, pixel_zw)).tolist()
    

    by_value.extend(zeros)
    by_mvalue.extend(mzeros)    

    imgd = np.asarray(by_value).reshape(-1, 1)
    im_idx = np.asarray(by_mvalue).reshape(-1, 3)
    
    #print(imgd)

    try:
        bgmm = BayesianGaussianMixture(n_components=3, covariance_type='full', max_iter=1000 ,n_init= 10, weight_concentration_prior_type='dirichlet_process').fit(imgd)

    except ValueError:
        #Not enough pixels in the box
        return

    clusterm = np.zeros(rgb_image.shape, rgb_image.dtype)

    labels = np.zeros(rgb_image.shape[:2], rgb_image.dtype)
    print(labels.shape)

    for index, cluster_no in enumerate(bgmm.predict(imgd)):
        
        #Retrieve x and y in the original matrix 
        depth, height, width = by_mvalue[index]

        #Color based on cluster 
        r,g,b = plt.cm.get_cmap('Set3')(cluster_no, bytes=True)[:3]#.to_rgba(bytes=True)[:3]
      

        clusterm[int(height), int(width)]= np.array([r,g,b], dtype=np.uint8) #[r,g,b] #cm(cluster_no)[:3] #(r,g,b)
        labels[int(height), int(width)]= cluster_no
 
    
    return clusterm, labels

def PILresize(imgpath, basewidth=416):

    img = Image.open(imgpath)
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    
    return img.resize((basewidth, hsize), PIL.Image.ANTIALIAS)
    


def find_bboxes(rgb_image, out, thr=0.1):

    img = lightnet.Image.from_bytes(open(out, 'rb').read())

    #Object detection threshold defaults to 0.1 here
    #img = lightnet.Image(rgb_image.astype(np.float32)) #DOCUMENTED BUT ACTUALLY MESSES UP A LOT!!!
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
        x_ = x #*tgt_w 
        y_ = y #*tgt_h
        w_ = w #*tgt_w
        h_ = h #*tgt_h
        
        print("Scaled: (%s, %s, %s, %s)" % (x_,y_,w_,h_))
        #And change coord system for later cropping
        x1 = (x_ - w_/2) #/ow
        y1 = (y_ - h_/2) #/oh
        x2 = (x_ + w_/2) #/ow
        y2 = (y_ + h_/2) #/oh

        
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
    return [(rgb_image[y1:y2, x1:x2], depthi[y1:y2, x1:x2]) for (x1,y1,x2,y2) in boxs_coord]


def max_bound(coord_list):

    areas=[]
  
    for coord in coord_list:
        
        w = coord[2] - coord[0]
        h = coord[3] - coord[1]
        a = w*h
        areas.append(a)

    max_a = max(areas)
    
    return [coord_list[k] for k, a in enumerate(areas) if a == max_a]
    

def rosimg_fordisplay(img_d):

    #Define same colormap as the histograms 
    cm = plt.cm.get_cmap('plasma')

    n, bins, patches = plt.hist(img_d.ravel(), 'auto', [0,10], color='green', edgecolor='black', linewidth=1)
    #Cutoff values will be based on bin edges
    thresholds = bins
    
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    # scale values to interval [0,1]
    col = bin_centers - min(bin_centers)
    col /= max(col)
    
    #print(img_d)

    new_d = np.zeros((img_d.shape[0],img_d.shape[1],3), img_d.dtype)
    for i, (c,p) in enumerate(zip(col, patches)):
         
        r, g, b = cm(c, bytes=True)[:3]
        #gray = 0.2989 * r + 0.5870 * g + 0.1140 * b 
        #print(r,g,b)
        color = r* 65536 + g*256 + b
        if i!=0:
            
            n= i-1
 
            #if thresholds[n]!=0.0:
    
            #img_d[np.logical_and(img_d>= thresholds[n], img_d <thresholds[i])] = color #gray
            lists = np.where(np.logical_and(img_d>= thresholds[n], img_d <thresholds[i]))    
            stacked = np.column_stack((lists[0], lists[1]))
            #print(stacked)
            for height, width in stacked:
                new_d[height, width] = np.array([r,g,b], dtype=np.uint8) 

   
    #new_d = cv2.normalize(new_d, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC3) 
    #print(new_d) 
    return new_d 

def backgr_sub(depth_image):
   
 
    n, bins, patches = plt.hist(depth_image.ravel(), 'auto', [1,10], color='green', edgecolor='black', linewidth=1)


    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    cm = plt.cm.get_cmap('gray')
    
    # scale values to interval [0,1]
    col = bin_centers - min(bin_centers)
    col /= max(col)


    area_dict ={}    
    #count_dict['counts']= []
    #Group bin edges in couples     
    cbins = chunks(bins)
    #print(bins)

    for i, edge in enumerate(cbins):

        if i==0:
   
            #Initialize
            start_p = edge[0] 
            end_p  = start_p + 1    

            area_dict[(start_p, end_p)]= 0

        if n[i] ==0.:
   
            #Skip empty bins
            continue

        if edge[0]<= end_p: #and edge[1] <= end_p:

            #Increase tot bin area in that interval
            area_dict[(start_p, end_p)] += (edge[1]-edge[0])*n[i]
                                    
        else:
        #elif edge[0]> end_p:
            #Update interval boundaries
            start_p = edge[0]
            end_p = start_p +1
 
            #And increase tot area with first one
            area_dict[(start_p, end_p)] = (edge[1]-edge[0])*n[i]

                    
    #print(area_dict)

    area_ord = sorted(area_dict.iteritems(), key=lambda (k,v): (v,k), reverse=True)
    
    #print(area_ord[:3])

    try:
        (left, right), _ = area_ord[2]  
    #print(right)
    except:
        (left,right), _ = area_ord[len(area_ord) -1]


    depth_image[np.logical_or(depth_image > right, depth_image==0.0)] = 10.0  
    
    
    #print(depth_image)
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

    return depth_image

def mask_binarize(depth_image):

    depth_image[depth_image!=10.0] = 1.0
    depth_image[depth_image==10.0] = 0.0
    
    #print(depth_image)

    depth_image = cv2.cvtColor(depth_image, cv2.COLOR_GRAY2BGR)

    return depth_image.astype(np.uint8)


def merge_clust(graph, src, dst):

    '''
    Function used for merging similar areas derived from the RAG
    After applying the predifined similarity threshold

    Skimage always calls it through (graph, src, dst) so it has to be in that format
    
    Trial without extra computation (i.e., only based on threshold)

    This method computes the mean color of `dst`.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    '''
    
    graph.node[dst]['total color'] += graph.node[src]['total color']
    graph.node[dst]['pixel count'] += graph.node[src]['pixel count']
    graph.node[dst]['mean color'] = (graph.node[dst]['total color'] / graph.node[dst]['pixel count'])
    
    
    
def weight_merging(graph, src, dst, n):

    '''
    Function to compute the new weights for the nodes which are 
    adjacent to the merged node, to update. 

    The method expects that the mean color of `dst` is already computed.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    n : int
        A neighbor of `src` or `dst` or both.

    Returns
    -------
    data : dict
        A dictionary with the `"weight"` attribute set as the absolute
        difference of the mean color between node `dst` and `n`.
    '''

    diff = graph.node[dst]['mean color'] - graph.node[n]['mean color']
    diff = np.linalg.norm(diff)
    return {'weight': diff}


def weight_boundary(graph, src, dst, n):
    """
    Handle merging of nodes of a region boundary region adjacency graph.

    This function computes the `"weight"` and the count `"count"`
    attributes of the edge between `n` and the node formed after
    merging `src` and `dst`.


    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    n : int
        A neighbor of `src` or `dst` or both.

    Returns
    -------
    data : dict
        A dictionary with the "weight" and "count" attributes to be
        assigned for the merged node.

    """
    default = {'weight': 0.0, 'count': 0}

    count_src = graph[src].get(n, default)['count']
    count_dst = graph[dst].get(n, default)['count']

    weight_src = graph[src].get(n, default)['weight']
    weight_dst = graph[dst].get(n, default)['weight']

    count = count_src + count_dst
    return {
        'count': count,
        'weight': (count_src * weight_src + count_dst * weight_dst)/count
    }

def merge_boundary(graph, src, dst):
    pass


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
        grid_mat = []

        for bpath in bag_paths:    

            
            if bpath[-8:] == 'orig.bag':
                continue
            try:
                
                bag= rosbag.Bag(bpath)
                bridge = CvBridge()

                copy_iter =  list(bag.read_messages(topics=['/camera/rgb/image_raw', '/camera/depth/image_raw', '/camera/depth_registered/sw_registered/image_rect_raw']))
                #full_dlist = [(bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough"), str(msg.header.stamp.secs)+str(msg.header.stamp.nsecs), t) for topic, msg, t in bag.read_messages(topics=['/camera/depth/image_raw'])]
                for idx, (topic, msg, t) in enumerate(list(bag.read_messages(topics=['/camera/rgb/image_raw', '/camera/depth/image_raw', '/camera/depth_registered/sw_registered/image_rect_raw']))):
                
                    #print(topic)                                    
                    if topic =='/camera/rgb/image_raw':
                        
                        #RGB frame found
                        start_rgb = t 
                        rgb_msg = msg 
                        curr_i = idx
                        rgb_mat.extend([(bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough"), str(msg.header.stamp.secs)+str(msg.header.stamp.nsecs))])                         

                        #print(str(msg.header.stamp.secs)+str(msg.header.stamp.nsecs))
                        try:
                            lookup = copy_iter[curr_i:curr_i+5]
                        except:
                            #end of list reached
                            lookup = copy_iter[curr_i:len(copy_iter)-1]

                        for topic2, msg2, t2 in lookup:
                      
                            if topic2 == '/camera/depth_registered/sw_registered/image_rect_raw':
                                           
                                #Look for grid image when present
                                #Also keep link with timestamp of rgb (not all RGBs have grid associated with it)
                                #print(str(msg.header.stamp.secs)+str(msg.header.stamp.nsecs))

                                grid_mat.extend([(bridge.imgmsg_to_cv2(msg2, desired_encoding="32FC1"), str(msg2.header.stamp.secs)+str(msg2.header.stamp.nsecs), str(msg.header.stamp.secs)+str(msg.header.stamp.nsecs))])   
                                break                            
            
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
    '''
    print(len(grid_mat))
    print(len(rgb_mat))
    print(len(img_mat))
    

    '''

    #sys.exit(0)
    #Consistency check
    if abs(len(img_mat) - len(rgb_mat)) > 1:

        logging.info('Input sets do not match!')
        sys.exit(0)
   
    
    for k, (img,stamp) in enumerate(img_mat):

        rgb_img, rstamp = rgb_mat[k]
       
        tgt_res = (416, 312)
        
        out = os.path.join(args.outpath, '%s.png' % stamp)
 
        #Resize image to 416 x related height to work with YOLO  
        #Not efficient (writes to disk) but possibly more robust to different resolutions
        #rgb_res = PILresize(out)     
        #rgb_res.save(out)
        #rgb_res = cv2.imread(out)
        rgb_res= rgb_img.copy() 
        
        imgd = np.uint32(img)
        imgd = imgd*0.001

        #Resize image to 416x312 to work with YOLO bbox detection 
        #rgb_res = cv2.resize(rgb_res, dsize=tgt_res, interpolation=cv2.INTER_CUBIC)

        cv2.imwrite(out, rgb_res)
        
        #Using Lightnet to extract bounding boxes 
        bboxes = find_bboxes(rgb_res, out)

        #Convert boxes back to original image resolution
        #And from YOLO format (center coord) to ROI format (top-left/bottom-right)
        n_bboxes = convert_boxes(bboxes, rgb_img.shape, tgt_res)        

        #Divide image up based on bboxes coordinates 
        obj_list = crop_img(rgb_res, n_bboxes, img.astype(np.float32))
        #Repeat the same on depth matrix too

        #print(imgd.shape)
        #img_d = rosimg_fordisplay(imgd)
        
        #Apply background subtraction based on depth values 
        '''
        imgd = backgr_sub(np.float32(imgd))

        '''
        '''
        plt.imshow(imgd, cmap = plt.get_cmap('gray'))
        plt.axis('off')
        plt.savefig(os.path.join('/mnt/c/Users/HP/Desktop/KMI/bboxes/test.png'), bbox_inches='tight')
        plt.clf()
        '''
        '''
        imgd = mask_binarize(imgd)
        #print(rgb_img.dtype)
        #print(imgd.dtype)
        
        redImg = np.zeros(rgb_img.shape, rgb_img.dtype)
        redImg[:,:] = (0, 0, 255)
        #print(redImg.shape)
        #print(redImg.dtype)
        #print(imgd.shape)
        #print(imgd.dtype)

        redMask = cv2.bitwise_and(redImg.astype(np.uint8), redImg, mask=cv2.cvtColor(imgd, cv2.COLOR_BGR2GRAY))
        new_i = cv2.addWeighted(redMask, 1, rgb_img, 1, 0, rgb_img)

        #mask_out = cv2.subtract(rgb_img, imgd)
        '''

        #cv2.imwrite(out, new_i)

        #Find bounding box with max area
        #(Assuming it is a good proxy of closest foreground objects)
        #for no, (x_top, y_top, x_btm, y_btm) in enumerate(n_bboxes):
        #print(max_bound(n_bboxes))
        
        #x_top, y_top, x_btm, y_btm = max_bound(n_bboxes)[0]
        #print(x_top, y_top, x_btm, y_btm)
 
        #cv2.imwrite(out, rgb_res[y_top:y_btm, x_top:x_btm])
        #sys.exit(0)

        #for grid_img, st, st2 in grid_mat:
        #    print(st2)
            
        '''
        try:
            grid_img = [grid_img for grid_img, st, st2 in grid_mat if st2==rstamp][0]

        except Exception as e:
            print(str(e))
            grid_img = None

        print(rstamp)
        print(stamp)
        print(grid_img)
        #print(img)   
        
        sys.exit(0)
        
        '''
        try:
            for no, (x_top, y_top, x_btm, y_btm)  in enumerate(max_bound(n_bboxes)):
           
                #Break if it was not in previous video too
                if not os.path.isfile(out.split('.png')[0]+'_%s.png' % str(no)):
                    sys.exit(0)
 
                nimgd = imgd.copy()            
       
                #print("(%s, %s, %s, %s)" % (x_top,y_top,x_btm,y_btm))
                cv2.rectangle(rgb_img,(x_top, y_top),(x_btm,y_btm),(0,255,0),3) 
                #nimgd = imgd[130:140, 130:200]
                #nrgb = rgb_res[130:140, 130:200]
       
                nimgd = imgd[y_top:y_btm,x_top:x_btm]
                nrgb = rgb_res[y_top:y_btm, x_top:x_btm]

                #print(str(x_btm - x_top))            
                #print(str(y_btm - y_top))
                print(nimgd.shape)
            
                print(nrgb.shape)
                #print(rgb_res)
                #print(rgb_res.shape)
                #sys.exit(0)
                logging.info('Starting pixel-level clustering')

                #Depth + pixel height clustering
                #new_rgb, label_matrix  = BGM_Dirichlet(nimgd, nrgb)     
                #Trying to cluster only based on depth
                try:
                    new_rgb, label_matrix = BGM_Dirichlet(nimgd, nrgb)
                except TypeError:

                    continue 
                ''' 
                #Check if grid viz is present
                if grid_img is not None:
 
                    logging.info('Slicing grid image too')
                    ngrid = grid_img[y_top:y_btm, x_top:x_btm]
                    grid_array = np.array(ngrid, dtype = np.dtype('f8'))
                    #grid_array = np.array(ngrid, dtype = np.dtype('f8'))
                    #Normalize in the [0, 255] space
                    grid_norm = cv2.normalize(grid_array, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
                    #print(grid_norm.shape) 
                    #And go from 1 channel to 3
                    grid_norm = cv2.cvtColor(grid_norm, cv2.COLOR_GRAY2BGR)                 
                    #print(grid_norm.shape) 
                    #cv2.imwrite(out.split('.png')[0]+'_grid_%s.png' % str(no) , grid_norm)
                
                    #Overlay to the existing RGB (clustering-segmented)
                    new_rgb= cv2.addWeighted(ngrid, 0.3, new_rgb, 1, 0, new_rgb)
                '''
                sliced_image = rosimg_fordisplay(nimgd)     

                #plt.imshow(sliced_image, cmap = plt.get_cmap('plasma'))        
                #plt.savefig(out.split('.png')[0]+'dmap_%s.png' % str(no) )
                #plt.clf()
        
                #sliced_image = cv2.imread(out.split('.png')[0]+'dmap_%s.png' % str(no))
                #sliced_image = cv2.cvtColor(sliced_image, cv2.COLOR_BGR2RGB) 
                print(nrgb.shape)
                print(sliced_image.shape)
                print(new_rgb.shape)

                final = np.hstack((nrgb, sliced_image, new_rgb))
                

                cv2.imwrite(out.split('.png')[0]+'2feat_%s.png' % str(no) , final)
                 
                ################### RAG + Cluster Merging ###################################################
                '''
                #rag = graph.rag_mean_color(nrgb, label_matrix)
                edge_map = filters.sobel(color.rgb2gray(nrgb))
                rag = graph.rag_boundary(label_matrix, edge_map)

                plt.clf() #In case there were histograms from before
                viz = graph.show_rag(label_matrix, rag, nrgb, img_cmap= 'gray', edge_cmap='viridis')

                #Save resulting viz locally
                cbar = plt.colorbar(viz)
                plt.savefig(out.split('.png')[0]+'_rag.png', bbox_inches='tight')
                plt.clf()
            
                #sys.exit(0)
                new_labels = graph.merge_hierarchical(label_matrix, rag, thresh=0.02, rag_copy=False,
                                   in_place_merge=True,
                                   merge_func=merge_boundary,
                                   weight_func=weight_boundary)
                #new_labels = graph.cut_threshold(label_matrix, rag, thresh=30, in_place=True)
                #new_labels = graph.merge_hierarchical(label_matrix, rag, thresh=50, rag_copy=False,                
                           #in_place_merge=True, merge_func=merge_clust, weight_func=weight_merging) 
                #cv2.imwrite(out.split('.png')[0]+'_rag.png', viz) 
                #sys.exit(0)

                #nviz = graph.show_rag(new_labels, rag, nrgb, img_cmap= 'gray', edge_cmap='viridis')


                #Save resulting viz locally
            
                #cbar = plt.colorbar(nviz)
                #plt.savefig(out.split('.png')[0]+'_new_rag.png', bbox_inches='tight')

                #plt.clf()

                #Pastel-like coloring with averages, alternative option for overlay is possible too
                final = color.label2rgb(new_labels, nrgb, kind='avg')

                #final = segmentation.mark_boundaries(final, new_labels, (0, 0, 0))
                #Saving final segmented image
                cv2.imwrite(out.split('.png')[0]+'final_%s.png' % str(no) , final)
                '''
            #sys.exit(0)
            '''       
            #Do pre-processing steps within each bounding box        
            for obj, dimg in obj_list:
            
                cv2.imwrite('/mnt/c/Users/HP/Desktop/test_crop.png', obj)

                cv2.imwrite('/mnt/c/Users/HP/Desktop/depth_crop.png', dimg)
            '''
        except Exception as e:

            logging.info(str(e))
            continue

    logging.info("Complete...took %f seconds" % float(time.time() - start))
