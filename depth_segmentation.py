import os
import time
import cv2
import argparse 
from PIL import Image
import numpy as np
import sys

def contour_det(gray_img):

    #Binarize 
    ret,thresh = cv2.threshold(gray_img,127,255,cv2.THRESH_BINARY)    #+cv2.THRESH_OTSU)    
    
    #Find contours 
    #retrieves all of the contours and reconstructs a full hierarchy of nested contours, no approx
    __, contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) 

    return contours
   

def canny_edge(gray_img):

    return cv2.Canny(gray_img, 100, 105)


def draw_cnt(contours, img, fname, outpath):

    for cnt in contours:
        
        #Approximate polygons         
        epsilon = 0.1*cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt,epsilon,True)

        cv2.drawContours(img, [approx], -1, (0, 0, 255), 3)
        cv2.imwrite(os.path.join(outpath, fname), img)

    return

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("imgpath", help="Provide path to your depth images")
    parser.add_argument("outpath", help="Provide path to output imgs with drawn contours")
    args =parser.parse_args()

    start = time.time()

    files = os.listdir(args.imgpath)
    img_mat =[]
        
    for f in files:
        
        if f[-3:] == 'ini':
            #Skip config files, if any
            continue
        

        try:
            #img_array1 = Image.open(os.path.join(args.imgpath,f))   #.convert('L')
            #img_np = np.asarray(img_array1)

            img_array = cv2.imread(os.path.join(args.imgpath,f), cv2.IMREAD_UNCHANGED)
        
        except Exception as e:
            
            print("Problem while opening img %s" % f)
            print(str(e))
            
            #Skip corrupted or zero-byte images
            continue
            
        try:    

            #img = cv2.imdecode(img_np, cv2.IMREAD_UNCHANGED)

            #if img_array.shape ==3:
            #Just flattening the channels, looks like no info loss
                #print("test")
            gray = cv2.cvtColor(img_array,cv2.COLOR_BGR2GRAY)
            #else:
            #gray = img_array            

            edges = canny_edge(gray)

            #Segment
            contours = contour_det(edges)
            #cv2.imwrite('/mnt/c/Users/HP/Desktop/test.png', edges)
            #print(type(edges))
            #sys.exit(0)
            
            draw_cnt(contours, img_array, f, args.outpath)
            #cv2.imwrite('/mnt/c/Users/HP/Desktop/test.png', gray)
            #cv2.imwrite('/mnt/c/Users/HP/Desktop/test_ng.png', img_array)
            
            #sys.exit(0) 

        except Exception as e:

            print("Problem while processing %s" % f) 
            print(str(e))


        #Display, with segmentation

        img_mat.append(img_array)
          

    #Dump to compressed output for future re-use
    print(len(img_mat))    
    #print(img_mat[0].shape)

    np.save("./depth_collected.npy", img_mat)
 
    print("Complete...took %f seconds" % float(time.time() - start))
