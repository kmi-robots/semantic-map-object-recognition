"""
Converts ROS messages to OpenCV images
and back

"""
# Added online link with camera sensor
import rospy
import message_filters
from actionlib import SimpleActionServer
from dh_interaction.msg import explorationAction, explorationResult
from sensor_msgs.msg import Image, PointCloud2

from cv_bridge import CvBridge, CvBridgeError
from collections import OrderedDict
import tf
from std_srvs.srv import SetBool, SetBoolResponse
import cv2
from collections import Counter
import json
import os

from test import test, img_processing_pipeline
from DH_integration import DH_img_send, DH_status_send
from spatial import update_KB
from data_loaders import BGRtoRGB
from segment import white_balance
# from spatial import map_semantic, extract_SR, pcl_processing_pipeline


class ImageConverter:

    def __init__(self, path_to_input, args, model, device, base_trans, via_data=None):

        self.img = None
        self.VG_data, self.embedding_space, self.cardinalities, self.COLORS, self.all_classes = test(args.it, path_to_input, args, model, device, base_trans)
        self.path_to_input = path_to_input
        self.model = model
        self.args = args
        self.device = device
        self.base_trans = base_trans
        self.via_data = via_data
        self.im_publisher = rospy.Publisher("/camera/rgb/image_bbox", Image, queue_size=1)
        # second publisher after knowledge-based correction
        self.corrim_publisher = rospy.Publisher("/camera/rgb/image_corrected", Image, queue_size=1)
        self.bridge = CvBridge()
        self.action = SimpleActionServer("exploration", explorationAction, execute_cb=self.exploration_callback, auto_start=False)
        self.tf_lis = tf.TransformListener()
        self.area_DB = {}
        self.area_ID = "activity_0"
        self.pcl_processed = {self.area_ID: {}}
        self.pcl_processed[self.area_ID]["floor"] = []
        self.pcl_processed[self.area_ID]["other"] = []

        if not os.path.isfile("./SR_KB.json"):
            print("Initializing spatial rel KB")
            self.SR_KB = OrderedDict()
            # Detect walls and surfaces and add them to permanent db
            # Hardcoded for now
            self.SR_KB["global_rels"] = []
        else:
            print("Retrieving spatial rel KB")
            # prior global relations are loaded from local file instead
            with open("./SR_KB.json", 'r') as jf:

                self.SR_KB = json.load(jf)

        # self.SR_KB[self.area_ID] = {}
        # self.SR_KB[self.area_ID]["planar_surfaces"] = {}

    def exploration_callback(self, goal):

        if goal.mode == 0:
            im_subscriber = message_filters.Subscriber("/camera/rgb/image_raw", Image)
            d_subscriber = message_filters.Subscriber("/camera/depth/image_raw", Image)
            pcl_subscriber = message_filters.Subscriber("/camera/depth/points", PointCloud2)

            # synchronise three topics
            ts = message_filters.ApproximateTimeSynchronizer([im_subscriber, pcl_subscriber, d_subscriber],
                                                             queue_size=1, slop=0.1)
            # one callback for all
            ts.registerCallback(self.callback)

            res, stat_id = DH_status_send("Starting to look around", first=True)
            if not res.ok:
                print("Failed communication with Data Hub ")
                print(res.content)

            rospy.sleep(goal.duration)

            im_subscriber.unregister()
            pcl_subscriber.unregister()
            d_subscriber.unregister()

            if self.args.stage == "only-segment":
                # save VIA-formatted annotated JSON for all segmented images
                with open("./"+self.via_data["_via_settings"]["project"]["name"]+".json", 'w') as jf:
                    json.dump(self.via_data, jf)

            res, stat_id = DH_status_send("Stopping observation", first=True)

            if not res.ok:
                print("Failed communication with Data Hub ")
                print(res.content)

            self.action.set_succeeded(explorationResult(True))

        if goal.mode == 1:
            print("Saving observations locally...")

            with open("./SR_KB.json", 'a') as jf:
                json.dump(self.SR_KB, jf)

            print("Saved as ./SR_KB.json")

            # Eventually, empty area DB and observation counter
            self.area_DB = {}
            self.action.set_succeeded(explorationResult(True))

    def callback(self, img_msg, pcl_msg, depth_msg):

        cam_trans, _ = self.tf_lis.lookupTransform('base_footprint', 'camera_link', rospy.Time(0))

        data = OrderedDict()
        try:
            data["data"] = self.bridge.imgmsg_to_cv2(img_msg, 'bgr8')
            # uint16 depth values in mm
            data["depth_image"] = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1")

            assert isinstance(pcl_msg, PointCloud2)
            data["pcl"] = pcl_msg
            # self.points = point_cloud2.read_points(pcl_msg, field_names=("x","y","z"), skip_nans=False)

        except CvBridgeError as e:
            print(e)

        data["filename"] = str(img_msg.header.stamp.to_sec())
        data["regions"] = None

        # self.SR_KB["global_rels"]

        # self.SR_KB[data["filename"]] = []

        """Uncomment for planar surface extraction
        # extract surfaces from PCL and locate them too

        self.pcl_processed = pcl_processing_pipeline(self.pcl, self.pcl_processed, self.area_ID, cam_trans)


        if self.pcl_processed[self.area_ID]:
            # There are some annotated points recognised as floor

            floor_points = set(self.pcl_processed[self.area_ID]["floor"])

            #Update spatial KB
            self.SR_KB[self.area_ID]["planar_surfaces"]["floor"]= {

                                                                "coords": floor_points
                                                     }
        """
        self.pcl_processed[self.area_ID]["floor"] = []
        self.pcl_processed[self.area_ID]["other"] = []

        try:

            # Get robot latest location
            trans, _ = self.tf_lis.lookupTransform('map', 'base_link', rospy.Time(0))

            data["x"] = trans[0]
            data["y"] = trans[1]
            data["z"] = trans[2]
            # Find area ID based on robot location?
        except:
            # if available
            data["x"] = 0
            data["y"] = 0
            data["z"] = 0

        # Send a status message
        res, stat_id = DH_status_send("Sending new image from camera", first=True)

        if not res.ok:
            print("Failed communication with Data Hub ")
            print(res.content)

        # send acquired img to Data Hub
        res, _, _, _ = DH_img_send(data, self.area_ID)

        if not res.ok:
            print("Failed to send img to Data Hub ")
            print(res.content)

        res, stat_id = DH_status_send("Analysing the image", status_id=stat_id)

        if not res.ok:
            print("Failed communication with Data Hub ")
            print(res.content)

        if self.args.stage == "only-segment":
            # save img locally
            save_img_path = self.via_data["_via_settings"]["core"]["default_filepath"]
            cv2.imwrite(os.path.join(save_img_path, data["filename"].replace('.', '_') + ".png"),
                        BGRtoRGB(white_balance(data["data"])))

            # process with segmentation only (without classifying)
            self.via_data = img_processing_pipeline(data, self.path_to_input, self.args, self.model, self.device,
                                                    self.base_trans, self.cardinalities, self.COLORS, self.all_classes,
                                                    self.args.K, self.args.sem, self.args.Kvoting, self.VG_data, [], [],
                                                    self.embedding_space, via_data=self.via_data)
            # skip the remainder
            return

        # Then images are processed one by one by calling run_processing_pipeline directly

        processed_data, _, _, _ = img_processing_pipeline(data, self.path_to_input, self.args, self.model, self.device,
                                                          self.base_trans, self.cardinalities, self.COLORS,
                                                          self.all_classes, self.args.K, self.args.sem,
                                                          self.args.Kvoting, self.VG_data, [], [], self.embedding_space)

        res, stat_id = DH_status_send("Image analysed", status_id=stat_id)

        if not res.ok:
            print("Failed communication with Data Hub ")
            print(res.content)

        # print(type(processed_img))
        # And publish results after processing the single image

        try:
            # self.im_publisher.publish(self.bridge.cv2_to_imgmsg(processed_data[0],'bgr8'))
            # self.corrim_publisher.publish(self.bridge.cv2_to_imgmsg(processed_data[1], 'bgr8'))

            data["data"] = processed_data[0]
            data["regions"] = processed_data[2]
            data["colours"] = processed_data[3]
            data["locations"] = processed_data[4]
            data["hsv_colours"] = processed_data[5]

            # Send processed image to Data Hub

            res, stat_id = DH_status_send("Sending processed image", status_id=stat_id)
            if not res.ok:
                print("Failed communication with Data Hub ")
                print(res.content)

            res, res_array, xyz_img, self.area_DB = DH_img_send(data, self.area_ID, all_bins=self.area_DB)

            # Given results formatted as in DH, update internal KB for that particular frame
            # self.SR_KB["global_rels"][data["filename"]] = update_KB(self.SR_KB["global_rels"][data["filename"]], res_array, cam_trans)

            self.SR_KB = update_KB(data["filename"], self.SR_KB, res_array, cam_trans)

            if not res.ok:
                print("Failed to send img to Data Hub ")
                print(res.content)

            self.im_publisher.publish(self.bridge.cv2_to_imgmsg(xyz_img, 'rgb8'))

            print("Dictionary contains %i processed frames now" % len(self.SR_KB["global_rels"]))  # .keys()))
            # print(self.SR_KB)
            # Optional TO-DO: sends a third image after knowledge-based correction

        except CvBridgeError as e:
            print("The provided image could not be processed")
            print(e)

    def start(self, rate):
        self.action.start()
        while not rospy.is_shutdown():
            # to make sure it publishes at x Hz
            rate.sleep()




