import rospy
import requests
from requests.auth import HTTPBasicAuth
import datetime
import cv2
import base64
import os
import json


# from matplotlib.colors import rgb2hex

#---- Hardcoded DH API params ------------#
teamid = "kmirobots"
teamkey = "0e920c05-e7a0-4745-9e94-eff7e1343b5d"

url = "https://api.mksmart.org/sciroc-competition/kmirobots/"

imgtype = "ImageReport"
sttype = "RobotStatus"
format = "image/jpeg"
episode = "EPISODE12"
#-----------------------------------------#



def DH_img_send(img_obj, area_id, all_bins={}):

    """
    :param img_obj:  expects a dictionary object with img + metadata
    :return: string with status
    """

    t = datetime.datetime.fromtimestamp(rospy.Time.now().secs)
    timestamp = t.strftime("%Y-%m-%dT%H:%M:%SZ")

    img_id  = img_obj["filename"].replace(".",'_')
    xyz_img = img_obj["data"]

    results = []
    r = 0.05  # 5 cm


    if img_obj["regions"] is not None and img_obj["regions"]!=[]:

        #This is the captured original image, i.e., pre annotation
        img_id = img_id + "_processed"

        #find the position of each object based on depth map

        labels, scores, coords, ranks = zip(*img_obj["regions"])

        locs = img_obj["locations"]

        #center_coords = [(int(x+(x2-x)/2), int(y+ (y2-y)/2))  for x,x2,y,y2 in coords]

        #read all points in pointcloud
        #points_list = point_cloud2.read_points_list(pcl, field_names=("x", "y", "z")) #, uvs=center_coords)

        for i, obj_label in enumerate(labels): #obj_label, score, coords, rank  in img_obj["regions"]:

            x,y,x2,y2 = coords[i]

            vertices_x, vertices_y, vertices_z = locs[i]
            vertices_coords = list(zip(vertices_x, vertices_y, vertices_z))

            ranking_list = [{'item': key, 'score': val} for key, val in ranks[i].items()]

            #adding colour coding
            #but swapping from open cv's default BGR back to RGB for DH web GUI
            bgr_array =img_obj["colours"][i].tolist()
            
            colour_array = [bgr_array[2], bgr_array[1], bgr_array[0]] #/255

            node = {"item": obj_label,
                    "score": scores[i],
                    "colour_code": colour_array,
                    "ranking": ranking_list,
                    "box_tl": (x,y),
                    "box_br": (x2,y2),
                    "box_area": (x2-x)*(y2-y),
                    "centre_coords": vertices_coords[0],
                    "box_vertices_coords": vertices_coords[1:]
                    }

            results.append(node)

            # print(node)


            """Uncomment to map observations to associated bin/grid
            
            map_x, map_y, map_z = vertices_coords[0]
            # create a sphere for that point:
            # of center (cx, cy, cz) and radius r

            if map_x is not None: #non-empty location

                if not all_bins: # if dict empty

                    all_bins[area_id] = {}
                    all_bins[area_id][(map_x, map_y, map_z, r)] = []
                    all_bins[area_id][(map_x, map_y, map_z, r)].append(node)

                else:


                    try:
                        for i, (cx, cy, cz, r) in enumerate(list(all_bins[area_id].keys())):

                            if (map_x - cx) **2 + (map_y - cy) **2 + (map_z - cz) **2 < r **2:
                                # Check if object is already in existing bin
                                all_bins[area_id][(cx, cy, cz, r)].append(node)

                                break

                            else:

                                 if i == len(list(all_bins[area_id].keys()))-1: #if last iter

                                    # not present yet, create new bin and add point to it
                                    all_bins[area_id][(map_x, map_y, map_z, r)]= []
                                    all_bins[area_id][(map_x, map_y, map_z, r)].append(node)

                    except ValueError:

                        print("There was a problem iterating through the provided area DB")
                        print(all_bins[area_id].keys())
            # And draw center coords on img
            # cv2.circle(xyz_img, (u,bot_y), 5, img_obj["colours"][i], thickness=5, lineType=8, shift=0)
            #cv2.putText(xyz_img, "( "+str(map_x)+", "+str(map_y) + ", "+str(map_z)+" )", (u-10, v-10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour_array*255, 2)
            """
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


    return requests.request("POST", complete_url, data=json.dumps(json_body),auth=HTTPBasicAuth(teamkey, '')), results, xyz_img, all_bins


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

