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

def preproc(imgm, img_wd):
        

    cv2.imwrite('/mnt/c/Users/HP/Desktop/KMI/out-canny/depth-asis.png',imgm)
    
    
    #for record in img:
    i=0
   
    img_new =[] 

    for row in img_wd:
        
        for j in range(len(row)):
            if row[j] > 1.0:

                img_new[i,j]=0.0    
                #cutoff threshold
            else:

                img_new[i,j]= imgm[i,j]
                #copy original value
        i+=1 
    
    cv2.imwrite('/mnt/c/Users/HP/Desktop/KMI/out-canny/depth-masked.png',img_new)
    sys.exit(0)

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    #gray = np.uint8(gray)   
    #ret, markers=  cv2.connectedComponents(gray)
    edges = canny_edge(gray)

    kernel = np.ones((10,10),np.uint8)

    #op =  cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    
    closing =  cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    
    #dist_transform = cv2.distanceTransform(gray,cv2.DIST_L2,5)
    jet = cv2.applyColorMap(closing,cv2.DIST_L2,5)

    maxIntensity = 255.0 # depends on dtype of image data
    x = arange(maxIntensity) 

    # Parameters for manipulating image data
    phi = 2.0
    theta = 2.0

    y = (maxIntensity/phi)*(x/(maxIntensity/theta))**0.5

    '''
    hist,bins = np.histogram(closing.flatten(),256,[0,256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max()/ cdf.max()
    plt.plot(cdf_normalized, color = 'b')
    plt.hist(closing.flatten(),256,[0,256], color = 'r')
    plt.xlim([0,256])
    plt.legend(('cdf','histogram'), loc = 'upper left')
    plt.show()
    '''
    equ = cv2.equalizeHist(closing)
    res = np.hstack((closing,equ)) #stacking images side-by-side
    cv2.imwrite('/mnt/c/Users/HP/Desktop/KMI/out-canny/res-aligned_'+f,res)

    # Decrease intensity such that
    # dark pixels become much darker, 
    # bright pixels become slightly dark 

    closingc = (maxIntensity/phi)*(equ/(maxIntensity/theta))**2
    closingc = array(closingc,dtype=uint8)
    #blur = cv2.bilateralFilter(closing,5,20,20)

    closing =  cv2.morphologyEx(equ, cv2.MORPH_CLOSE, kernel)
    #cv2.imwrite('/mnt/c/Users/HP/Desktop/KMI/out-canny/blurred_'+f, blur) 
    #kernel = np.ones((5,5),np.uint8)
    op = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    #cv2.imwrite('/mnt/c/Users/HP/Desktop/KMI/out-canny/notcontrasted'+f, closing) 
    #cv2.imwrite('/mnt/c/Users/HP/Desktop/KMI/out-canny/contrasted'+f, closingc) 
    
    canny= cv2.Canny(op, 80, 85)

    jet = cv2.applyColorMap(equ,cv2.DIST_L2,5)
    cv2.imwrite('/mnt/c/Users/HP/Desktop/KMI/out-canny/jet_equ'+f, jet)
    #cv2.imwrite('/mnt/c/Users/HP/Desktop/KMI/out-canny/canny_'+f, canny) 
    __, contours,hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    cv2.drawContours(jet, contours, -1, (0, 0, 255), 3)

    cv2.imwrite('/mnt/c/Users/HP/Desktop/KMI/out-canny/jet_cont'+f, jet)
    #sys.exit(0)
    #edges=gray
    #print(cv2.connectedComponents(gray, 4, cv2.CV_32S))
    #cv2.imwrite('/mnt/c/Users/HP/Desktop/KMI/test',cv2.connectedComponents(gray, 4, cv2.CV_32S))
    
    return edges

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
        img_mat =[bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough") for topic, msg, t in bag.read_messages()]
        bag.close()
    
        
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
        
                 img_mat.append(img_array)

            except Exception as e:
            
                print("Problem while opening img %s" % f)
                print(str(e))
            
                #Skip corrupted or zero-byte images
                continue

    for img in img_mat:

        #print(img.shape) 
        #Trying to convert back to mm from uint16
        width, height = img.shape        

        imgd = np.uint32(img)
        imgd = imgd*0.001
        
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

            edged =preproc(img, imgd)
            sys.exit(0)
            '''
            kernel = np.ones((5,5),np.uint8)
            #Segment
            contours = contour_det(edged)
            
            #dil = cv2.dilate(edged,kernel, iterations=2)
            #opening = cv2.morphologyEx(dil, cv2.MORPH_OPEN, kernel)

            gradient =  cv2.morphologyEx(edged, cv2.MORPH_GRADIENT, kernel)
            #cv2.imwrite('/mnt/c/Users/HP/Desktop/KMI/out-canny/'+f, gradient)
            #print(type(edges))
            sys.exit(0)
            
            poly_contours = draw_cnt(contours, img_array, f, args.outpath)
            findAndMatch(contours2=poly_contours, img2=img_array, fname=f)
            #cv2.imwrite('/mnt/c/Users/HP/Desktop/test.png', gray)
            #cv2.imwrite('/mnt/c/Users/HP/Desktop/test_ng.png', img_array)
            '''
            #sys.exit(0) 

        except Exception as e:

            print("Problem while processing ") 
            print(str(e))


        #Display, with segmentation

          

    #Dump to compressed output for future re-use
    print(len(img_mat))    
    #print(img_mat[0].shape)

    np.save("./depth_collected.npy", img_mat)
 
    print("Complete...took %f seconds" % float(time.time() - start))
