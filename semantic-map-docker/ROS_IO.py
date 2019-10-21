"""
Converts ROS messages to OpenCV images
and back

"""

#Added online link with camera sensor
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge,CvBridgeError
from collections import OrderedDict
#import cv2

from test import test, run_processing_pipeline

class ImageConverter:


    def __init__(self, path_to_input, args, model, device, base_trans):

        self.img = None
        self.VG_data, self.embedding_space, self.cardinalities,self.COLORS, self.all_classes = test(args.it, path_to_input, args, model, device, base_trans)
        self.im_publisher = rospy.Publisher("/camera/rgb/image_bbox", Image, queue_size=1)
        self.corrim_publisher = rospy.Publisher("/camera/rgb/image_corrected", Image, queue_size=1) #second publisher after knowledge-based correction
        self.bridge = CvBridge()
        self.im_subscriber = rospy.Subscriber("/camera/rgb/image_raw", Image, self.callback)


    def callback(self, msg):

        try:

            self.img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.timestamp = msg.header.stamp.to_sec()

        except CvBridgeError as e:

            print(e)

    def start(self, path_to_input,args, model, device, base_trans, rate):

        while not rospy.is_shutdown():

            if self.img is not None:

                #show subscribed image
                #cv2.imshow('cam in',self.img)
                #cv2.waitKey(10000)
                #cv2.destroyAllWindows()

                #processed_imgs = test(args.it, path_to_input, args, model, device, base_trans, camera_img=(self.timestamp,self.img))

                data = OrderedDict()
                data["filename"] = str(self.timestamp)
                data["regions"] = None
                data["data"] = self.img

                #Then images are processed one by one by calling run_processing_pipeline directly

                processed_imgs, _, _, _ = run_processing_pipeline(data, path_to_input, args, model,  device, base_trans \
                                                                  , self.cardinalities,self.COLORS, self.all_classes, \
                                                              args.K, args.sem, args.Kvoting, self.VG_data, [], [], self.embedding_space)

                #print(type(processed_img))
                #And publish results after processing the single image
                try:

                    self.im_publisher.publish(self.bridge.cv2_to_imgmsg(processed_imgs[0],'bgr8'))
                    self.corrim_publisher.publish(self.bridge.cv2_to_imgmsg(processed_imgs[1], 'bgr8'))
                    #rate.sleep() #to make sure it publishes at 1 Hz

                except CvBridgeError as e:

                    print(e)


