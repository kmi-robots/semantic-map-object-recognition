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



def get_histogram(img_array, stampid):

    #Compute histogram of values row by row
    '''
    hist, bin_edges = np.apply_along_axis(lambda x: np.histogram(x, bins='auto'), 0, img_array)
    #print(hist)
    #print(bin_edges)
    #print(type(hist))
    plt.bar(bin_edges[:-1], hist,width=np.diff(bin_edges))
    '''
    
    cm = plt.cm.get_cmap('gray')
    print(cm)
    # Plot histogram.
    n, bins, patches = plt.hist(img_array.ravel(), 'auto', [0,10], color='green', edgecolor='black', linewidth=1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # scale values to interval [0,1]
    col = bin_centers - min(bin_centers)
    col /= max(col)
    
    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', cm(c)) 
        print(str(cm(c)[:3]))

    '''
    #print(img_array.dtype)
    #hist = cv2.calcHist([img_array], [0], None, [256], [0,256])
    plt.hist(img_array.ravel(),'auto',[0,10], cmap='gray') 
    #plt.plot(hist, color = 'b')
    '''
    plt.xlim([0,10])
    #plt.show()    
    if os.path.isfile(os.path.join('/mnt/c/Users/HP/Desktop/KMI/histograms', stampid+'.png')):
        print("skipping")
        return

    plt.savefig(os.path.join('/mnt/c/Users/HP/Desktop/KMI/histograms', stampid+'.png'))
    #plt.show()
    plt.clf()

    cm = plt.cm.get_cmap('gray')

    # Plot histogram.
    n, bins, patches = plt.hist(img_array.ravel(), 'auto', [1,10], color='green', edgecolor='black', linewidth=1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # scale values to interval [0,1]
    col = bin_centers - min(bin_centers)
    col /= max(col)
    
    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', cm(c)) 

    '''
    #print(img_array.dtype)
    #hist = cv2.calcHist([img_array], [0], None, [256], [0,256])
    plt.hist(img_array.ravel(),'auto',[0,10], cmap='gray') 
    #plt.plot(hist, color = 'b')
    '''
    #print(n)
    #sys.exit(0)
    plt.xlim([1,10])
    plt.ylim([1,30000])
    #plt.show()    
    plt.savefig(os.path.join('/mnt/c/Users/HP/Desktop/KMI/histograms', stampid+'_noout.png'))
    plt.clf()

    return os.path.join('/mnt/c/Users/HP/Desktop/KMI/histograms', stampid+'.png')

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

def rosimg_fordisplay(img_array, img_d, stampid):

    #Define same colormap as the histograms 
    cm = plt.cm.get_cmap('gray')

    n, bins, patches = plt.hist(img_d.ravel(), 'auto', [0,10], color='green', edgecolor='black', linewidth=1)

    #Cutoff values will be based on bin edges
    thresholds = bins
    
    '''
    for i, th in enumerate(thresholds):
 
        if i!=0:


        else:
            img_d[img_d < th] = 
    '''
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    # scale values to interval [0,1]
    col = bin_centers - min(bin_centers)
    col /= max(col)
    
    print(img_d)
    for i, (c,p) in enumerate(zip(col, patches)):
        
        r, g, b = cm(c)[:3]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        #print(gray)
        #sys.exit(0)
        if i!=0:

            
            n= i-1 
            #print(gray)
            #print(thresholds[n])  
            #print(thresholds[i]) 
            if thresholds[n]!=0.0:
                img_d[np.logical_and(img_d>= thresholds[n], img_d <thresholds[i])] = gray


            #np.where(np.logical_and(img_d>= thresholds[n], img_d <thresholds[i]), gray, imgd)
            #plt.setp(p, 'facecolor', cm(c)) 
            #print(str(cm(c)[:3]))
        #else:
            
            #sys.exit(0) 
            #print(thresholds[i])                      
            #img_d[img_d <= thresholds[i]] = gray

    
    return img_d


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

def clustering(image, K):

    cluster_model = KMeans(init='k-means++', n_clusters=K)    
    cluster_model.fit(image)
    
    return (cluster_model.labels_, cluster_model.cluster_centers_)


if __name__ == '__main__':

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

        print("Loading rosbag...")
        
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
        
                
        print("Resolution of your input images is "+str(width)+"x"+str(height))        

        #print(img[100])        
        
        '''
        img =np.uint32(img)
        img = img*0.001

        print(np.max(img))
        print(np.min(img))
        sys.exit(0)   
        '''
        
        if not os.path.isfile(os.path.join('/mnt/c/Users/HP/Desktop/KMI/histograms-comp-00', stamp+'.png')):
            #print("skip")
            #continue

            try:    

                #img = cv2.imdecode(img_np, cv2.IMREAD_UNCHANGED)
                #params = preproc(img, imgd, stamp)

                #print(stamp)
            
                '''
                cv_image_norm = cv2.normalize(img, img, 0, 1, cv2.NORM_MINMAX)
    
                cv_image_norm = cv_image_norm.astype(np.uint8)
                #print(cv_image_norm.shape)     
                '''

                #Uncomment to create histograms for the first time
                #get_histogram(np.float32(imgd), stamp)
           
                #print(np.float32(imgd)) 
                sliced_image = rosimg_fordisplay(img, np.float32(imgd),stamp)
                #print(sliced_image)
                
                
                plt.imshow(sliced_image, cmap = plt.get_cmap('tab20b'))
                
                #plt.show()
                plt.savefig(os.path.join('/mnt/c/Users/HP/Desktop/KMI/histograms-comp-00', stamp+'.png'))
                plt.clf()
                #cv2.imwrite( sliced_image)
                #sys.exit(0)
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
            

                #sys.exit(0)
            
                '''

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
                #sys.exit(0)

        #Display, with segmentation
        '''
        img_orig = cv2.imread(os.path.join('/mnt/c/Users/HP/Desktop/KMI/histograms-comp', stamp+'.png'))

        hst = cv2.imread(os.path.join('/mnt/c/Users/HP/Desktop/KMI/histograms', stamp+'_noout.png'))
        #plot =cv2.cvtColor(hst, cv2.COLOR_BGR2GRAY)
            
        #print(plot.shape)
        #print(img_orig.shape)

        K=5
        labels, centroids = clustering(img, K)
        
        print(labels)
        print(centroids)
        
        sliced_image = rosimg_fordisplay(img, np.float32(imgd),stamp)
         
        plt.imshow(sliced_image, cmap = plt.get_cmap('gray'))        
        plt.savefig(os.path.join('/mnt/c/Users/HP/Desktop/KMI/out-canny/cont/_orig%s.png' % stamp))
        plt.clf()
        
        print(sliced_image.dtype)    
   
        sliced_image = cv2.imread(os.path.join('/mnt/c/Users/HP/Desktop/KMI/out-canny/cont/_orig%s.png' % stamp))
        sliced_image =cv2.cvtColor(sliced_image, cv2.COLOR_BGR2GRAY)
        __, contours,hierarchy = cv2.findContours(sliced_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        cv2.drawContours(sliced_image, contours, -1, (0, 0, 255), 3)

        cv2.imwrite('/mnt/c/Users/HP/Desktop/KMI/out-canny/cont/_cont%s.png' % stamp, sliced_image)
        
        sys.exit(0)
        '''
        #Uncomment for image, histogram aligned output
        #aligned = np.hstack((img_orig, hst))
        #cv2.imwrite(os.path.join('/mnt/c/Users/HP/Desktop/KMI/histograms-comp', stamp+'.png'), aligned)
        
          

    #Dump to compressed output for future re-use
    print(len(img_mat))    
    #print(img_mat[0].shape)

    np.save("./depth_collected.npy", img_mat)
 
    print("Complete...took %f seconds" % float(time.time() - start))
