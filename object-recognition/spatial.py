import math
from sensor_msgs import point_cloud2
import struct
from collections import Counter
import statistics as stat

import pandas as pd
from pyntcloud import PyntCloud
import open3d
import numpy as np


from common_sense import formatlabel

###### All thresholds used for spatial reasoning####################

""" Vertices of bbox named as:

           tl#-----tm#-----tr#
             |               |
           ml#      c#     mr#
             |               |
           bl#-----bm#-----br#

"""

wall_th = 0.05 # 5 cm
cm_tol = 0.05
wall_syn = "wall.n.01" # same synset used on VG (same case as table, avoid ambiguity)
floor_syn = "floor.n.01"
ring_r = 0.5 # 50 cm

def find_real_xyz(xtop, ytop, xbtm, ybtm, pcl, RGB_RES=(480,640)):

    #our u,v in this case are the coords of the center of each bbox
    u = int(xtop + (xbtm - xtop) / 2)
    v = int(ytop + (ybtm - ytop) / 2)
    #and scale to depth image resolution (640*480)
    # u = int(round(u/RGB_RES[1] * D_RES[1]))
    # v = int(round(v/RGB_RES[0] * D_RES[0]))

    #Equivalent of center coords in pointcloud
    # map_x, map_y, map_z = pixelTo3DPoint(pcl, u, v)

    bot_y = int(ybtm)
    top_y = int(ytop)
    top_x = int(xtop)
    bot_x = int(xbtm)

    """Handle overflowing bounding boxes"""

    if bot_y >= RGB_RES[0]:

        bot_y = RGB_RES[0] - 5

    if  top_y <= 0:

        top_y += 5

    if top_x <= 0:

        top_x +=5

    if bot_x >= RGB_RES[1]:

        bot_x = RGB_RES[1] - 5

    #order: c, br, tl, ml, bl, bm, mr, tr, tm
    vertices = [(u, v), (bot_x, bot_y), (top_x, top_y), (top_x, v), (top_x, bot_y), (u, bot_y),
                (bot_x,v), (bot_x, top_y), (u,top_y)]

    vertices_x, vertices_y, vertices_z = list(zip(* point_cloud2.read_points_list(\

                    pcl, field_names=("x","y","z"), skip_nans=False, uvs=vertices)))

    vertices_x = [None if math.isnan(x_val) else x_val for x_val in vertices_x]
    vertices_y = [None if math.isnan(y_val) else y_val for y_val in vertices_y]
    vertices_z = [None if math.isnan(z_val) else z_val for z_val in vertices_z]

    return vertices_x, vertices_y, vertices_z


def pixelTo3DPoint(cloud, u, v):

    #Equivalent results as library read_points
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
                "centre_coords": bin_[:-1],
                "box_vertices_coords": group[0]["box_vertices_coords"]

            }

        except KeyError:

            semmap[area_id] = {}
            semmap[area_id][bin_] = {

                "prediction": obj_pred,
                "abs_freq": freq,
                "tot_obs": len(group),
                "med_score": med_score,
                "centre_coords": bin_[:-1],
                "box_vertices_coords": group[0]["box_vertices_coords"]
            }

    return semmap


def pcl_processing_pipeline(pointcloud, preproc_pointcloud, area_ID, cam_trans):

    pc_list = point_cloud2.read_points_list(pointcloud, skip_nans=True, field_names=("x", "y", "z"))

    points = pd.DataFrame(pc_list, columns=["x", "y", "z"])
    cloud = PyntCloud(points)

    cloud.add_scalar_field("plane_fit", max_dist=1e-2, max_iterations=150)

    binary_planes = cloud.points['is_plane'].to_numpy(copy=True)

    xyz = cloud.points[['x', 'y', 'z']].to_numpy(copy=True)

    for i in np.argwhere(binary_planes):   #find indices of non-zero values

        pl_x = xyz[i, :][0][0]
        pl_y = xyz[i, :][0][1]
        pl_z = xyz[i, :][0][2]

        # Add plane annotation to pcl logged data as well

        if pl_y <= cam_trans[1]: #camera height wrt to robot base
            #"floor"
            preproc_pointcloud[area_ID]["floor"].append((pl_x, pl_y, pl_z))

        else:

            # type 2 surface, i.e., "other"
            preproc_pointcloud[area_ID]["other"].append((pl_x, pl_y, pl_z))

    """
    if __debug__:
        # Expensive I/O, active only in debug mode

        plane_colors = np.zeros((binary_planes.shape[0], 3))
        plane_colors[binary_planes == 0] = [255, 0, 127]  # acquamarine if not planar
        plane_colors[binary_planes == 1] = [0, 0, 0]  # black if planar

        cloud.to_file('./temp_pcl.ply')

        # import point_cloud_utils as pcu
        # v, _, _, _ = pcu.read_ply("my_model.ply")
        # n = pcu.estimate_normals(n, k=16)

        open3d.utility.set_verbosity_level(open3d.utility.VerbosityLevel.Debug)

        pcd = open3d.read_point_cloud("./temp_pcl.ply")  # temp_pcl.ply")  # Read the point cloud

        pcd.colors = open3d.Vector3dVector(plane_colors)  # binary_planes)
        # pcd.colors open3d.utility.Vector3dVector
        
        open3d.draw_geometries([pcd])
        #print(preproc_pointcloud[area_ID])
    """

    return preproc_pointcloud


def extract_SR(semmap, area_ID, SR_KB):

    """
    :param semmap: semantic map at specific t
    :param Sr_KB: expects dictionary of prior spatial  found over time
    also includes floor points, if any was found
    :param area_ID: index of specific area under observation to subset data
    :return: updated set of spatial relations between objects
    """

    fl_pset = set()

    if SR_KB[area_ID]["planar_surfaces"]:

        fl_pset = SR_KB[area_ID]["planar_surfaces"]["floor"]["coords"]

    obj_l = list(semmap[area_ID].keys())

    #iterate in reverse order to not mess up as items are removed
    for j, bin_ in reversed(list(enumerate(obj_l))):

        obj = semmap[area_ID][bin_]
        lab, syn = formatlabel(obj["prediction"])


        # order: c, br, tl, ml, bl, bm, mr, tr, tm
        br_coords, tl_coords, ml_coords, bl_coords, bm_coords, mr_coords, tr_coords, tm_coords =obj["box_vertices_coords"]


        #object-floor relationship
        if fl_pset:

            try:

                #Estimate base plane
                p1 = np.array(bl_coords)
                p2 = np.array(bm_coords)
                p3 = np.array(br_coords)

                # These two vectors are in the plane
                v1 = p3 - p1
                v2 = p2 - p1
                # the cross product is a vector normal to the plane
                cp = np.cross(v1, v2)
                a, b, c = cp
                # This evaluates a * x3 + b * y3 + c * z3 which equals d
                d = np.dot(cp, p3)

                #print('The equation is {0}x + {1}y + {2}z = {3}'.format(a, b, c, d))
                box_base = []
                box_base.extend(br_coords)
                box_base.extend(bl_coords)
                box_base.extend(bm_coords)

                # Add more random points belonging to that plane, with x,y in +5-5 cm from bl and br
                br_x, br_y, _ = br_coords
                bl_x, bl_y, _ = bl_coords
                rand_x = np.random.uniform(low=float(bl_x - cm_tol), high= float(br_x + cm_tol), size=(50,) ).tolist()
                rand_y = np.random.uniform(low=float(bl_y - cm_tol), high= float(br_y + cm_tol), size=(50,) ).tolist()

                # ax + by + cz + d = 0
                # z = (-d -ax - by)/c
                box_base.extend([(rand_x, rand_y, float((-d -a*rand_x - b*rand_y)/c)) \
                                 for rand_x, rand_y in list(zip(rand_x, rand_y)) ])

                #Extract all xyz points in obj bbox base

                #Find intersection with floor set
                if list(set(box_base) & fl_pset):

                    #If any point intersects, then estimate that object is on floor
                    try:
                        SR_KB["global_rels"]["ON(("+syn.name()+","+lab+"),("+floor_syn+","+"floor))"] +=1

                    except KeyError:

                        #first time this rel is found
                        SR_KB["global_rels"]["ON(" + syn.name() + "," + floor_syn + ")"] = 1

            except TypeError:

                #If any of the coord values is None we just skip it for now
                #TODO find best estimate for points when one or more is None
                continue

        # object-object relationships
        #Ring calculus as in Young et al. for now
        cx, cy, cz = obj["centre_coords"]
        all_others = [bin_ for k, bin_ in reversed(list(enumerate(obj_l))) if k!=j]

        for binn in all_others:

            centre_x, centre_y, centre_z = binn[:-1]

            if (centre_x - cx) ** 2 + (centre_y - cy) ** 2 + (centre_z - cz) ** 2 < ring_r ** 2:

                obj2 = semmap[area_ID][bin_]
                lab2, syn2 = formatlabel(obj2["prediction"])

                try:

                    SR_KB["global_rels"]["NEAR(("+syn.name()+","+lab+"),("+syn2.name()+","+lab2+"))"] +=1
                    #add bi-directional
                    SR_KB["global_rels"]["NEAR((" + syn2.name() + "," + lab2 + "),(" + syn.name() + "," + lab + "))"] += 1

                except KeyError:
                    SR_KB["global_rels"][
                        "NEAR((" + syn.name() + "," + lab + "),(" + syn2.name() + "," + lab2 + "))"] = 1
                    # add bi-directional
                    SR_KB["global_rels"][
                        "NEAR((" + syn2.name() + "," + lab2 + "),(" + syn.name() + "," + lab + "))"] = 1

        #remove object from list to not repeat comparisons
        obj_l.pop(j)


    return SR_KB
