import rospy
import requests
from requests.auth import HTTPBasicAuth
from sensor_msgs.msg import PointCloud2
import datetime
import cv2
import base64
import os
import json
from sensor_msgs import point_cloud2
import struct
import math
from matplotlib.colors import rgb2hex

#---- Hardcoded DH API params ------------#
teamid = "kmirobots"
teamkey = "0e920c05-e7a0-4745-9e94-eff7e1343b5d"

url = "https://api.mksmart.org/sciroc-competition/kmirobots/"

imgtype = "ImageReport"
sttype = "RobotStatus"
format = "image/jpeg"
episode = "EPISODE12"
#-----------------------------------------#



def DH_img_send(img_obj):

    """
    :param img_obj:  expects a dictionary object with img + metadata
    :return: string with status
    """

    t = datetime.datetime.fromtimestamp(rospy.Time.now().secs)
    timestamp = t.strftime("%Y-%m-%dT%H:%M:%SZ")

    img_id  = img_obj["filename"].replace(".",'_')
    pcl = img_obj["pcl"]
    xyz_img = img_obj["data"]

    RGB_RES = xyz_img.shape  #e.g., 480x640x3

    results = []


    if img_obj["regions"] is not None and img_obj["regions"]!=[]:

        #This is the captured original image, i.e., pre annotation
        img_id = img_id + "_processed"

        #find the position of each object based on depth map

        labels, scores, coords, ranks = zip(*img_obj["regions"])
        #center_coords = [(int(x+(x2-x)/2), int(y+ (y2-y)/2))  for x,x2,y,y2 in coords]

        #read all points in pointcloud
        #points_list = point_cloud2.read_points_list(pcl, field_names=("x", "y", "z")) #, uvs=center_coords)

        for i, obj_label in enumerate(labels): #obj_label, score, coords, rank  in img_obj["regions"]:

            x,y,x2,y2 = coords[i]
            
            #our u,v in this case are the coords of the center of each bbox
            u = int(x + (x2 - x) / 2)
            v = int(y + (y2 - y) / 2)

            #and scale to depth image resolution (640*480)
            # u = int(round(u/RGB_RES[0] * D_RES[0]))
            # v = int(round(v/RGB_RES[1] * D_RES[1]))

            #Equivalent of center coords in pointcloud
            
            # map_x, map_y, map_z = pixelTo3DPoint(pcl, u, v)

            #Handle overflowing boxes
            bot_y = int(y2)

            if y2 >= RGB_RES[0]:

                bot_y = RGB_RES[0] - 5


            (map_x, base_x), (map_y, base_y), (map_z, base_z) = list(zip(* point_cloud2.read_points_list(\

                            pcl, field_names=("x","y","z"), skip_nans=False, uvs=[(u,v), (u,bot_y)])))

            if math.isnan(map_x):
                map_x = None
            if math.isnan(map_y):
                map_y = None
            if math.isnan(map_z):
                map_z = None

            # Same for position of base of bbox (later used wrt floor)

            if math.isnan(base_x):
                base_x = None
            if math.isnan(base_y):
                base_y = None
            if math.isnan(base_z):
                base_z = None


            ranking_list = [{'item': key, 'score': val} for key, val in ranks[i].items()]

            #adding colour coding
            #but swapping from open cv's default BGR back to RGB for DH web GUI
            bgr_array =img_obj["colours"][i].tolist()
            
            colour_array = [bgr_array[2], bgr_array[1], bgr_array[0]] #/255

            # rgb2hex(colour_array)


            node = {'item': obj_label,
                    'score': scores[i],
                    'colour_code': colour_array,
                    'ranking': ranking_list,
                    'box_top': (x,y),
                    'box_bottom': (x2,y2),
                    'map_x': map_x,
                    'map_y': map_y,
                    'map_z': map_z,
                    'bbase_x': base_x,
                    'bbase_y': base_y,
                    'bbase_z': base_z
                    }

            results.append(node)

            # print(node)

            # And draw center coords on img
            # cv2.circle(xyz_img, (u,bot_y), 5, img_obj["colours"][i], thickness=5, lineType=8, shift=0)
            #cv2.putText(xyz_img, "( "+str(map_x)+", "+str(map_y) + ", "+str(map_z)+" )", (u-10, v-10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour_array*255, 2)

    complete_url = os.path.join(url,"sciroc-episode12-image", img_id)

    #Convert img array to base64

    base64_img = arrayTo64(img_obj["data"])

    json_body = { "@id": img_id,
                  "@type": imgtype,
                  "team" : teamid,
                  "timestamp": timestamp,
                  "x": img_obj["x"],
                  "y": img_obj["y"],
                  "z": img_obj["z"],
                  "base64": base64_img,
                  "format": format,
                  "results": results,
                }


    return requests.request("POST", complete_url, data=json.dumps(json_body),auth=HTTPBasicAuth(teamkey, '')), xyz_img


def DH_status_send(msg, status_id="", first=False):

    """
    :param msg:
    :param first: this boolean flag adds increments for messages at the same timestamp
    :return:
    """
    t = datetime.datetime.fromtimestamp(rospy.Time.now().secs)
    timestamp = t.strftime("%Y-%m-%dT%H:%M:%SZ")

    if first:
        #create the first onecloud
        status_id = "status_" + timestamp.replace(":",'')+"_1"
    else:

        i = int(status_id.split('_')[-1]) + 1
        status_id ="_".join(status_id.split('_')[:-1]) + "_"+str(i)


    complete_url = os.path.join(url, "sciroc-robot-status", status_id)

    json_body = {"@id": status_id,
                 "@type": sttype,
                 "message": msg,
                 "episode": episode,
                 "team": teamid,
                 "timestamp": str(timestamp),
                 "x": 0,
                 "y": 0,
                 "z": 0
                 }

    return requests.request("POST", complete_url, data=json.dumps(json_body),auth=HTTPBasicAuth(teamkey, '')), status_id

def arrayTo64(img_array):

    """
    :param img_array: expects img in array format read through OpenCV
    :return: base64 converted stringcloud
    """
    _, buffer = cv2.imencode('.jpg', img_array)

    return base64.b64encode(buffer).decode('utf-8')

def pixelTo3DPoint(cloud, u, v):

    width = cloud.width
    height = cloud.height
    point_step = cloud.point_step
    row_step = cloud.row_step

    array_pos = v*row_step + u*point_step

    bytesX = [x for x in cloud.data[array_pos:array_pos+4]]
    bytesY = [x for x in cloud.data[array_pos+4: array_pos+8]]
    bytesZ = [x for x in cloud.data[array_pos+8:array_pos+12]]

    byte_format = struct.pack('4B', *bytesX)
    X = struct.unpack('f', byte_format)[0]

    byte_format = struct.pack('4B', *bytesY)
    Y = struct.unpack('f', byte_format)[0]

    byte_format = struct.pack('4B', *bytesZ)
    Z = struct.unpack('f', byte_format)[0]

    return float(X), float(Y), float(Z)

