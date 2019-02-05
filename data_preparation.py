#################################################
##
## Simple script to read ShapeNet and/or NYUDepth samples 
## as vecs and save ground truth labels as well
## in the npy format to use them as test set 
## with the Siamese-like implementation
##
################################################

import sys, os 
import numpy as np

from shape_match import mainContour, cropToC, random_pick_from
import cv2

import time

#from broader_pipeline import find_bboxes, convert_boxes

from itertools import combinations, product 

##########
# Select the input set to generate couples from: google-search, shapenet, nyu or kmi
#
#
##########

#No. of examples to randomly pick from each folder
SP= 200

# no of positive examples for each class  = n! / r!(n-r)!
pos_labels = int(SP*(SP-1)) 
neg_labels = int(SP**2)
tot= pos_labels + neg_labels

## Trial: init empty containers first to lower memory burden
a_ = np.empty((tot, 160, 60, 3))
b_ = np.empty((tot, 160, 60, 3))
labels_ = np.empty((tot, 2))

master_idx = 0

#Optional, just for debugging 
def display_img(img):

    from PIL import Image

    formatted = img.astype('uint8')
    img = Image.fromarray(formatted, 'RGB')
    img.save('/home/agni/test.png')
    img.show()
    #sys.exit(0)


def img_preproc(path_to_image):

    img = cv2.imread(path_to_image)
    x = cv2.resize(img, (60,160))
    x = np.asarray(x)
    #display_img(x)
    x = x.astype('float')
    #display_img(x)
    x = np.expand_dims(x, axis= 0)

    #display_img(x)
    return x


def update_sets(idx_pairs, images, neg=False):     #, master_idx = 0):

    global master_idx
    global a_, b_

    for idx_left, idx_right in idx_pairs:

       label = labels_[master_idx:master_idx] 
       if neg:

           img_left= images[0][idx_left] 
           img_right= images[1][idx_right] 
           label[:] = np.array([1,0])

       else:
           img_left= images[idx_left] 
           img_right= images[idx_right]
           label[:] = np.array([0,1])


       #Updating larger containers defined upfront with slicing 
       slice_a = a_[master_idx:master_idx+1]
       slice_b = b_[master_idx:master_idx+1]
       slice_a[:] = img_left
       slice_b[:] = img_right

       #start = master_idx
       master_idx +=1 
       
       '''
       if master_idx is None:

           # Single slice of main container
           slice_a = a_[:1]
           print(a_[0:1].shape)
           sys.exit(0)
           left_a[:] = img_left  
           
           slice_b = b_[:1]
           slice_b[:] = img_right
           
           master_idx +=1
           #init= False

       else:
             a.append(img_left) #np.vstack((a,img_left))
             b.append(img_right) #np.vstack((b,img_right))

       '''
    #How many examples were added?
    #return (master_idx-start)
    #return a, b


if __name__ == "__main__":
    
    
    if sys.argv[1] == 'google-search':


        # e.g., path to chair examples
        train_examples1 =[os.path.join(sys.argv[2], img) for img in os.listdir(sys.argv[2])]
        
        # e.g., path to plant examples
        train_examples2 =[os.path.join(sys.argv[3], img) for img in os.listdir(sys.argv[3])]
        
        #isfirst = True

        
        train_examples1 = random_pick_from(train_examples1, sizepick=SP)
        train_examples2 = random_pick_from(train_examples2, sizepick=SP)
        
        t1 = time.time()

        #path_pairs1 = combinations(train_examples1, 2)
        # e.g., positive examples for chair
        objectclass1 = [img_preproc(img1) for img1 in train_examples1]       
        #pos_pairs1 = combinations(objectclass1, 2)        
        indices1 = [idx for idx in range(len(objectclass1))]

        update_sets(combinations(indices1, 2), objectclass1)        
        #a_ = np.stack(a)
        #a, b = update_sets(pos_pairs1)

        #print(a.shape)
        
        print("First combo: %f seconds " % float(time.time() - t1)) 
        # e.g., positive examples for plant
        objectclass2 = [img_preproc(img2) for img2 in train_examples2]
        indices2 = [idx for idx in range(len(objectclass2))]

        pos_pairs2 = combinations(indices2, 2)

        print("Second combo %f seconds " % float(time.time() - t1)) 

        update_sets(pos_pairs2, objectclass2)

        print("Updated a and b in %f seconds " % float(time.time() - t1)) 
        #pos_labels = [np.array([1,0]) for i in range(sampleno + sampleno2)]
        #labels = np.stack(pos_labels)
        
        print("Pos labels in %f seconds " % float(time.time() - t1)) 
        #negative examples, e.g., chairs vs plants
        
        neg_pairs = product(indices1, indices2)

        
        print("Product in %f seconds " % float(time.time() - t1))

        update_sets(neg_pairs,[objectclass1, objectclass2], neg=True)
        #neg_labels = [np.array([0,1]) for i in range(neg_sampleno)]
        #neg_labels = np.stack(neg_labels)

        #a = np.stack(a)
        '''
        a  = a + nega
        b = b + negb
        a = np.stack(a)
        b = np.stack(b)
        '''
        #labels = pos_labels + neg_labels 
        #labels = np.stack(labels)

        print(a_.shape)
        print(b_.shape)
        print(labels_.shape)
        

        print("Completed in %f seconds " % float(time.time() - t1)) 
        print("The generated set contains %i examples" % int(a_.shape[0]) )
        print("With %i positive examples" % pos_labels)
        print("And %i negative examples" % neg_labels)


        #sys.exit(0)
        np.savez_compressed('../google-derived/twoclass_train_val_set.npz', a=a_, b=b_, labels=labels_)
        #np.savez_compressed('../google-derived/imgset_right.npz', b_)
        #np.savez_compressed('../google-derived/gt_labelset.npz', labels_)

        print("Data sets and label sets saved to disk")

