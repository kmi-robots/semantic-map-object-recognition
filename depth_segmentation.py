import os
import cv2
import argparse 
from PIL import Image
import numpy as np





if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("imgpath", help="Provide a path to your depth images")
    args =parser.parse_args()

    start = time.time()

    files = os.listdir(args.imgpath)
    img_mat =[]
        
    for f in files:


        try:
            img_array1 = Image.open(os.path.join(args.imgpath,f))   #.convert('L')
            img_np = np.asarray(img_array1)
            img_mat.append(img_np)
          
        except Exception as e:
            
            print("Problem while opening img %s" % f)
            print(str(e))
            
            #Skip corrupted or zero-byte images
            continue

    #Dump to compressed output for future re-use
        
    print(img_mat.shape)

    np.save("./depth_collected.npy", img_mat)
 
    print("Complete...took %f seconds" % float(time.time() - start))
