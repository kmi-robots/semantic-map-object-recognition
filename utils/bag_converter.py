"""
Python 2.6 script to manipulate images
captured as ROSbag messages
and saved them in pickled format
NOTE: bags in dir provided need to be indexed

In this example, messages are temporally sampled,
to reduce dimensionality and extract a more representative
test sample.

Change the window variable to play with the temporal window
If forLabels set to True then messages are saved as img files instead,
to be later tagged
"""

import rosbag
from cv_bridge import CvBridge
import argparse
import os
import time
from PIL import Image
import numpy as np
import cv2

window = 5.0 # Every 5 seconds
forLabels = True

def main(path_to_bag):

    start = time.time()
    print("Loading rosbag...")

    img_mat =[]

    for bagf in os.listdir(path_to_bag):

        print(bagf)
        bag = rosbag.Bag(os.path.join(path_to_bag, bagf))
        bridge = CvBridge()

        timestamp = 0.0
        
        for topic, msg, _ in bag.read_messages():

            #Filter depth data or other topics in case any other was collected
            if topic != '/camera/rgb/image_raw':
                continue

            current = msg.header.stamp.to_sec()

            if forLabels:

                cv2.imwrite(os.path.join(os.environ['HOME'],'OpenLabeling/main/input/img_'+bagf+'_'+str(current)+'.png'),\
                             bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough"))
                continue

            if (current - timestamp) > window:

                img_mat.append((bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough"),\
                    str(current)))

                timestamp = current

        bag.close()

        print("Rosbag imported, took %f seconds" % float(time.time() - start))

    print("Sampled %i images in total" % len(img_mat))

    print("Saving all imported images as numpy pickle")

    np.save('../semantic-map-docker/data/robot_collected.npy', img_mat)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('bag', help='Path to bag files to be converted.')
    args = parser.parse_args()

    main(args.bag)


