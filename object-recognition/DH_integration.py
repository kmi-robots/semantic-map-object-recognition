import rospy
import requests
from requests.auth import HTTPBasicAuth
import datetime
import cv2
import base64
import os
import json

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

    if img_obj["regions"] is not None:

        #This is the captured original image, i.e., pre annotation
        img_id = img_id + "_processed"

        #find the position of each object based on depth map
        dmap = img_obj["depth_map"]

        for obj_label, score, coords, rank  in img_obj["regions"]:

            x,x2,y,y2 = coords
            center_depth = dmap[int(x+(x2-x)/2), int(y+ (y2-y)/2)] #uint16 in mm

            #TODO: transform based on camera position and robot position

            #append to list of x,y,z object locations to send to DH together with object list

            continue

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
                  "object_list": img_obj["regions"]
                }


    return requests.request("POST", complete_url, data=json.dumps(json_body),auth=HTTPBasicAuth(teamkey, ''))


def DH_status_send(msg, status_id="", first=False):

    """
    :param msg:
    :param first: this boolean flag adds increments for messages at the same timestamp
    :return:
    """
    t = datetime.datetime.fromtimestamp(rospy.Time.now().secs)
    timestamp = t.strftime("%Y-%m-%dT%H:%M:%SZ")

    if first:
        #create the first one
        status_id = "status_" + timestamp.replace(":",'')+"_1"
    else:

        i = int(status_id.split('_')[-1]) + 1
        status_id ="_".join(status_id.split('_')[:-1]) + "_"+str(i)
        pass

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
    :return: base64 converted string
    """
    _, buffer = cv2.imencode('.jpg', img_array)

    return base64.b64encode(buffer).decode('utf-8')
