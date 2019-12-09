import math
from sensor_msgs import point_cloud2
import struct
from collections import Counter
import statistics as stat

from common_sense import formatlabel

###### All thresholds used for spatial reasoning####################

wall_th = 0.05 # 5 cm
wall_syn = "wall.n.01" # same synset used on VG (same case as table, avoid ambiguity)



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


def map_semantic(area_DB, area_id, semmap ={}):

    """
    :param area_DB: dict where observations in scouted area are grouped by location bin
    :param semmap: could optionally update a pre-existing semantic map
    :return: semantic map with aggregated predictions at certain location and median confidence score
    """

    # Then check observations across those 5 waypoints by bin/sphere
    for bin_, group in area_DB[area_id].items():

        votes = Counter([entry['item'] for entry in group])
        obj_pred, freq = votes.most_common(1)[0] # select most frequent one

        #Median of confidence score across all observations
        med_score = stat.median([entry['score'] for entry in group])

        try:
            semmap[area_id][bin_] = {

                "prediction": obj_pred,
                "abs_freq": freq,
                "tot_obs": len(group),
                "med_score": med_score,
                "map_coords": bin_[:-1],
                "bbase_coords": group[0]["bbase_coords"]
            }
        except KeyError:

            semmap[area_id] = {}
            semmap[area_id][bin_] = {

                "prediction": obj_pred,
                "abs_freq": freq,
                "tot_obs": len(group),
                "med_score": med_score,
                "map_coords": bin_[:-1],
                "bbase_coords": group[0]["bbase_coords"]
            }

    return semmap


def extract_SR(semmap, area_ID, SR_KB):

    """
    :param semmap: semantic map at specific t
    :param Sr_KB: expects dictionary of prior spatial  found over time
    :return: updated set of spatial relations between objects
    """

    #Relationships wrt planar surfaces
    try:

        for entry in SR_KB[area_ID]["planar_surfaces"]:

            #focus on ON(object, surface) type of relationships

            if entry["surface_type"] == "wall":

                wall_z = entry["coords"][-1]

                for bin_ in semmap[area_ID].keys():

                    if bin_[2] - wall_z <= wall_th:

                        #All objects leaning against wall / hanging on wall
                        obj_l, obj_syn = formatlabel (semmap[area_ID][bin_]["prediction"])

                        SR_KB["global_rels"]["ON( ("+obj_l+","+obj_syn+"),(wall,"+wall_syn+")"] +=1


            if entry["surface_type"] == "tabletop" or entry["surface_type"] == "floor":

                sur_x = entry["coords"][0]


    except KeyError:

        #new area
        pass

    #object-object relationships

    for bin_ in semmap[area_ID].keys():

        continue


    return SR_KB
