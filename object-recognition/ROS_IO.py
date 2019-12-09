"""
Converts ROS messages to OpenCV images
and back

"""

#Added online link with camera sensor
import rospy
import message_filters
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs import point_cloud2
import pcl
from cv_bridge import CvBridge,CvBridgeError
from collections import OrderedDict
import tf
from std_srvs.srv import SetBool,SetBoolResponse
#import cv2
from collections import Counter
import json
import os
import numpy as np
# import ros_numpy
import pandas as pd
from pyntcloud import PyntCloud
import open3d


from test import test, img_processing_pipeline
from DH_integration import DH_img_send, DH_status_send
from spatial import map_semantic, extract_SR


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
        self.obs_counter = 0
        self.area_DB = {} #OrderedDict()
        self.area_ID = "activity_0"


    def service_callback(self, msg):

        if msg.data:

            self.im_subscriber = message_filters.Subscriber("/camera/rgb/image_raw", Image) #, self.callback, queue_size=1)
            self.d_subscriber = message_filters.Subscriber("/camera/depth/image_raw", Image)
            self.pcl_subscriber = message_filters.Subscriber("/camera/depth/points", PointCloud2)

            #synchronise two topics
            self.ts = message_filters.ApproximateTimeSynchronizer([self.im_subscriber, self.pcl_subscriber, self.d_subscriber], queue_size=1, slop=0.1)
            #one callback for both
            self.ts.registerCallback(self.callback)

            res, stat_id = DH_status_send("Starting to look around", first=True)

            #Counter to keep track of spatial relations
            if not os.path.isfile("./SR_KB.json"):

                print("Initializing spatial rel KB")
                self.SR_KB = OrderedDict()
                #Detect walls and surfaces and add them to permanent db
                #Hardcoded for now

                self.SR_KB[self.area_ID] = {}
                self.SR_KB[self.area_ID]["planar_surfaces"] = []

                self.SR_KB["global_rels"] = Counter()


                self.SR_KB[self.area_ID]["planar_surfaces"].append({ "surface_type": "tabletop",

                                                  "coords": (0.0,0.0, 0.61)
                                                                })

                self.SR_KB[self.area_ID]["planar_surfaces"].append({"surface_type": "wall",

                                                 "coords": (0.30, 0.5, 2.5)
                                                 })

            else:

                print("Retrieving spatial rel KB")
                with open("./SR_KB.json", 'r') as jf:

                    self.SR_KB = json.load(jf)


            if not res.ok:

                print("Failed communication with Data Hub ")
                print(res.content)

            self.obs_counter += 1

            return SetBoolResponse(True, "Image subscriber registered")

        else:

            self.im_subscriber.unregister()
            self.pcl_subscriber.unregister()
            self.d_subscriber.unregister()


            if self.obs_counter >= 1: #e.g., stop and reason on scouted area every 5 waypoints

                #Current semantic map
                semantic_map_t0 = map_semantic(self.area_DB, self.area_ID)

                self.SR_KB = extract_SR(semantic_map_t0, self.area_ID, self.SR_KB)

                # Eventually, empty area DB and observation counter
                self.obs_counter = 0
                self.area_DB = {}


            res, stat_id = DH_status_send("Stopping observation", first=True)

            if not res.ok:
                print("Failed communication with Data Hub ")
                print(res.content)

            with open("./SR_KB.json", 'w') as jf:

                json.dump(self.SR_KB, jf)

            print("Saved updated SR KB locally")

            return SetBoolResponse(False,"Shutting down image subscriber")


    def callback(self, img_msg, pcl_msg, depth_msg):

        try:

            self.timestamp = img_msg.header.stamp.to_sec()
            self.img = self.bridge.imgmsg_to_cv2(img_msg, 'bgr8')

            self.dimg = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1") #uint16 depth values in mm


            assert isinstance(pcl_msg, PointCloud2)
            self.pcl = pcl_msg
            #self.points = point_cloud2.read_points(pcl_msg, field_names=("x","y","z"), skip_nans=False)


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
                data["pcl"] = self.pcl
                data["depth_image"] = self.dimg

                # TO-DO extract surfaces from PCL and locate them too

                pc_list = point_cloud2.read_points_list(self.pcl, skip_nans=True, field_names=("x", "y", "z"))

                points = pd.DataFrame(pc_list, columns=["x", "y", "z"])
                cloud = PyntCloud(points)


                #save temporary for 3D viz/debugging


                cloud.add_scalar_field("plane_fit")

                binary_planes = cloud.points['is_plane'].to_numpy(copy=True)

                plane_colors = np.zeros((binary_planes.shape[0], 3))
                plane_colors[binary_planes == 0] = [255, 0, 127] # acquamarine if not planar
                plane_colors[binary_planes == 1] = [0, 0, 0] #black if planar



                if __debug__:

                    #Expensive I/O, active only in debug mode

                    cloud.to_file('./temp_pcl.ply')

                    open3d.utility.set_verbosity_level(open3d.utility.VerbosityLevel.Debug)

                    pcd = open3d.read_point_cloud("./temp_pcl.ply")  # temp_pcl.ply")  # Read the point cloud

                    pcd.colors = open3d.Vector3dVector(plane_colors) #binary_planes)
                    # pcd.colors open3d.utility.Vector3dVector
                    open3d.draw_geometries([pcd])
                    os.remove('./temp_pcl.ply')

                """
                
                
                
                # cloud.plot()
                seg = cloud.make_segmenter()
                seg.set_model_type(pcl.SACMODEL_PLANE)
                seg.set_method_type(pcl.SAC_RANSAC)
                inds, plane_model = seg.segment()
                """
                self.img = None #to deal with unregistered subscriber
                self.pcl = None
                self.dimg = None

                try:

                    #Get robot latest location
                    trans, _ = self.tf_lis.lookupTransform('/map', '/base_link', rospy.Time(0))

                    data["x"] = trans[0]
                    data["y"] = trans[1]
                    data["z"] = trans[2]
                    # Find area ID based on robot location?

                except:

                    #if available
                    data["x"] = 0
                    data["y"] = 0
                    data["z"] = 0



                #Send a status message
                res,stat_id = DH_status_send("Sending new image from camera", first=True)

                if not res.ok:

                    print("Failed communication with Data Hub ")
                    print(res.content)

                #send acquired img to Data Hub
                res,_, _ = DH_img_send(data, self.area_ID)

                if not res.ok:
                    print("Failed to send img to Data Hub ")
                    print(res.content)

                res,stat_id = DH_status_send("Analysing the image", status_id=stat_id)

                if not res.ok:
                    print("Failed communication with Data Hub ")
                    print(res.content)

                #Then images are processed one by one by calling run_processing_pipeline directly

                processed_data, _, _, _ = img_processing_pipeline(data, path_to_input, args, model,  device, base_trans \
                                                                  , self.cardinalities,self.COLORS, self.all_classes, \
                                                              args.K, args.sem, args.Kvoting, self.VG_data, [], [], \
                                                                  self.embedding_space)
                #,VQA= True)



                # labs= list(zip(*processed_data[2]))[0]

                res, stat_id = DH_status_send("Image analysed",status_id=stat_id)

                if not res.ok:
                    print("Failed communication with Data Hub ")
                    print(res.content)

                #print(type(processed_img))
                #And publish results after processing the single image

                try:

                    #self.im_publisher.publish(self.bridge.cv2_to_imgmsg(processed_data[0],'bgr8'))

                    #self.corrim_publisher.publish(self.bridge.cv2_to_imgmsg(processed_data[1], 'bgr8'))
                    
                    data["data"] = processed_data[0]
                    data["regions"] = processed_data[2]
                    data["colours"] = processed_data[3]
                    data["locations"] = processed_data[4]

                    #Send processed image to Data Hub

                    res, stat_id = DH_status_send("Sending processed image", status_id=stat_id)
                    if not res.ok:

                        print("Failed communication with Data Hub ")
                        print(res.content)

                    res, xyz_img, self.area_DB = DH_img_send(data, self.area_ID, all_bins= self.area_DB)

                    #and also update local collection grouped by navigation area
                    #self.area_DB[self.cnt] = data    # Appended with incremental no.


                    if not res.ok:
                        print("Failed to send img to Data Hub ")
                        print(res.content)

                    self.im_publisher.publish(self.bridge.cv2_to_imgmsg(xyz_img,'bgr8'))
                    #Optional TO-DO: sends a third image after knowledge-based correction


                except CvBridgeError as e:
                    print("The provided image could not be processed")
                    print(e)

            rate.sleep() #to make sure it publishes at 1 Hz




