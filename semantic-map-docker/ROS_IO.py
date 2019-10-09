"""
Converts ROS messages to OpenCV images
and back

"""

#Added online link with camera sensor
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge,CvBridgeError
import cv2

class ImageConverter:


    def __init__(self):

        self.img = None
        self.im_publisher = rospy.Publisher("/image_raw", Image, queue_size=10)
        self.bridge = CvBridge()
        self.im_subscriber = rospy.Subscriber("/image_raw", Image, self.callback)


    def callback(self, msg):
        try:

            self.img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        except CvBridgeError as e:

            print(e)

    def start(self):

        while not rospy.is_shutdown():

            if self.img is not None:

                #show subscribed image
                cv2.imshow('cam in',self.img)
                cv2.waitKey(10000)
                cv2.destroyAllWindows()

                try:

                    self.im_publisher.publish(self.bridge.cv2_to_imgmsg(self.img,'bgr8'))

                except CvBridgeError as e:

                    print(e)


