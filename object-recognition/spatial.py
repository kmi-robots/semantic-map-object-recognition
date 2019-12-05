import math
from sensor_msgs import point_cloud2
import struct
from collections import Counter
import statistics as stat


def find_real_xyz(xtop, ytop, xbtm, ybtm, pcl, RGB_RES=(480,640)):

    #our u,v in this case are the coords of the center of each bbox
    u = int(xtop + (xbtm - xtop) / 2)
    v = int(ytop + (ybtm - ytop) / 2)

    #and scale to depth image resolution (640*480)
    # u = int(round(u/RGB_RES[1] * D_RES[1]))
    # v = int(round(v/RGB_RES[0] * D_RES[0]))

    #Equivalent of center coords in pointcloud
    # map_x, map_y, map_z = pixelTo3DPoint(pcl, u, v)

    #Handle overflowing boxes
    bot_y = int(ybtm)

    if ybtm >= RGB_RES[0]:

        bot_y = RGB_RES[0] - 5


    (map_x, base_x), (map_y, base_y), (map_z, base_z) = list(zip(* point_cloud2.read_points_list(\

                    pcl, field_names=("x","y","z"), skip_nans=False, uvs=[(u,v), (u,bot_y)])))


    map_x = None if math.isnan(map_x) else map_x # round(map_x,2)
    map_y = None if math.isnan(map_y) else map_y # round(map_y, 2)
    map_z = None if math.isnan(map_z) else map_z # round(map_z, 2)

    # Same for position of base of bbox (later used wrt floor)
    base_x = None if math.isnan(base_x) else map_x #round(base_x, 2)
    base_y = None if math.isnan(base_y) else map_y #round(base_y, 2)
    base_z = None if math.isnan(base_z) else map_z # round(base_z, 2)


    return (map_x, base_x), (map_y, base_y), (map_z, base_z)


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


def map_semantic(area_DB, semmap ={}):

    """
    :param area_DB: dict where observations in scouted area are grouped by location bin
    :param semmap: could optionally update a pre-existing semantic map
    :return: semantic map with aggregated predictions at certain location and median confidence score
    """

    # Then check observations across those 5 waypoints by bin/sphere
    for bin, group in area_DB.items():

        votes = Counter([entry['item'] for entry in group])
        obj_pred, freq = votes.most_common(1)[0] # select most frequent one

        #Median of confidence score across all observations
        med_score = stat.median([entry['score'] for entry in group])

        semmap[bin] = {

            "prediction": obj_pred,
            "abs_freq": freq,
            "tot_obs": len(group),
            "med_score": med_score,
            "xyz": bin[:-1],
            "bbase_coords": group[0]["bbase_coords"]
        }

    return semmap


def extract_SR(semmap, SR_KB):

    """
    :param semmap: semantic map at specific t
    :param Sr_KB: expects dictionary of prior spatial  found over time
    :return: updated set of spatial relations between objects
    """



    return SR_KB
