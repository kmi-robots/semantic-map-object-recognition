"""
Converts ROS messages to OpenCV images
and back

"""

#Added online link with camera sensor
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge,CvBridgeError
from collections import OrderedDict
import tf
from std_srvs.srv import SetBool,SetBoolResponse
#import cv2

from test import test, run_processing_pipeline
from DH_integration import DH_img_send, DH_status_send

class ImageConverter:


    def __init__(self, path_to_input, args, model, device, base_trans):

        self.img = None
        self.VG_data, self.embedding_space, self.cardinalities,self.COLORS, self.all_classes = test(args.it, path_to_input, args, model, device, base_trans)
        self.im_publisher = rospy.Publisher("/camera/rgb/image_bbox", Image, queue_size=1)
        self.corrim_publisher = rospy.Publisher("/camera/rgb/image_corrected", Image, queue_size=1) #second publisher after knowledge-based correction
        self.bridge = CvBridge()
        self.s = rospy.Service("start_exploration", SetBool, self.service_callback)
        self.im_subscriber = None
        self.tf_lis = tf.TransformListener()

    def service_callback(self, msg):

        if msg.data:

            self.im_subscriber = rospy.Subscriber("/camera/rgb/image_raw", Image, self.callback, queue_size=1)

            print(DH_status_send("Starting to look around").content)

            return SetBoolResponse(True, "Image subscriber registered")

        else:

            self.im_subscriber.unregister()

            print(DH_status_send("Stopping observation").content)

            return SetBoolResponse(False,"Shutting down image subscriber")



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
                self.img = None #to deal with unregistered subscriber

                try:

                    #Get robot latest location
                    trans, _ = self.tf_lis.lookupTransform('/map', '/base_link', rospy.Time(0))

                    data["x"] = trans[0]
                    data["y"] = trans[1]
                    data["z"] = trans[2]


                except:

                    #if available
                    data["x"] = 0
                    data["y"] = 0
                    data["z"] = 0

                #Send acquired img to Data Hub


                print(DH_status_send("Sending new image from camera").content)
                print(DH_img_send(data).content)

                print(DH_status_send("Analysing the image").content)
                #Then images are processed one by one by calling run_processing_pipeline directly

                processed_data, _, _, _ = run_processing_pipeline(data, path_to_input, args, model,  device, base_trans \
                                                                  , self.cardinalities,self.COLORS, self.all_classes, \
                                                              args.K, args.sem, args.Kvoting, self.VG_data, [], [], self.embedding_space)

                print(DH_status_send("Image analysed").content)
                #print(type(processed_img))
                #And publish results after processing the single image

                try:

                    self.im_publisher.publish(self.bridge.cv2_to_imgmsg(processed_data[0],'bgr8'))
                    self.corrim_publisher.publish(self.bridge.cv2_to_imgmsg(processed_data[1], 'bgr8'))

                    data["data"]= processed_data[0]
                    data["regions"] = processed_data[2]
                    #Send processed image to Data Hub
                    print(DH_status_send("Sending processed image with annotated objects").content)
                    print(DH_img_send(data).content)

                    #Optional TO-DO: sends a third image after knowledge-based correction

                except CvBridgeError as e:

                    print(e)

            rate.sleep() #to make sure it publishes at 1 Hz