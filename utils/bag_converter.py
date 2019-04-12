"""
Python 2.6 script to manipulate images
captured as ROSbag messages
and saved them in pickled format
NOTE: bags in dir provided need to be indexed

In this example, messages are temporally sampled,
to reduce dimensionality and extract a more representative
test sample.

Change the window variable to play with the temporal window
"""

import rosbag
from cv_bridge import CvBridge
import argparse
import os
import time
from PIL import Image
import numpy as np

window = 5.0 # Every 5 seconds

def main(path_to_bag):

    start = time.time()
    print("Loading rosbag...")

    img_mat =[]

    for bagf in os.listdir(path_to_bag):

        bag = rosbag.Bag(os.path.join(path_to_bag, bagf))
        bridge = CvBridge()

        timestamp = 0.0
        for topic, msg, _ in bag.read_messages():

            current = msg.header.stamp.to_sec()

            if (current - timestamp) > window:

                img_mat.append((bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough"),
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


