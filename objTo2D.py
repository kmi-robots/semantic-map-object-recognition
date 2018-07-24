import os
import sys
from matplotlib import pyplot as plt
import meshpy
import autolab_core
from perception import CameraIntrinsics, ObjectRender, RenderMode
from PIL import Image
import numpy as np 
from visualization import Visualizer2D as vis
import autolab_core.utils as utils
import logging
import time


logging.getLogger().setLevel(logging.INFO)
#TODO:
# scale to more images
# keep ids and output in sub-folders grouped by id
# Read 3D model
read_start= time.time()

mesh = meshpy.ObjFile('model_normalized.obj').read()
read_stop = time.time()
logging.info('Read took %.3f sec' %(read_stop-read_start))

sub_start= time.time()
stable_poses = mesh.stable_poses()
mesh = mesh.subdivide() #min_tri_length=0.01)
mesh.compute_vertex_normals()
sub_end = time.time()
logging.info('Pose division took %.3f sec' %(sub_end-sub_start))

d = utils.sqrt_ceil(len(stable_poses))
vis.figure(size=(16,16))

########CAMERA SETUP#################################################################
#intrinsic matrix values are derived from K matrix in ros message
#projection matrix is indicated as P in ros
# (ros has a 3x4 default for P so only the left 3x3 portion has to be considered)

cam_start = time.time()
#Camera Intrinsics - based on Kinect 
ci = CameraIntrinsics(

    frame='camera',
    fx = 594.0502248744873,
    fy = 597.6261068216613,
    cx = 324.761877545329,
    cy = 234.3706583714745,
    #K = np.array([[570.3422241210938, 0.0, 314.5], [ 0.0, 570.3422241210938, 235.5], [0.0, 0.0, 1.0]]),
    skew=0.0,
    height=480,
    width=640
)
#######################################################################################


'''
#Defining obj to camera poses
#z axis away from scene, x to right, y up

rt = autolab_core.RigidTransform(

    rotation= np.array([[0., 0., -1.], [0., 1., 0.],[1.,0., 0.]]),
    from_frame='camera',
    to_frame='obj'
)


rgb_l, depth_l = cam.images(mesh, [rt])
'''

cam_stop = time.time()
logging.info('Camera set up, took %.3f sec' %(cam_stop-cam_start))

rend_start = time.time()

for k, stable_pose in enumerate(stable_poses):
    
    #print(stable_pose)
    #sys.exit(0)
    # set resting pose
    T_obj_world = mesh.get_T_surface_obj(stable_pose.T_obj_table).as_frames('obj', 'world')

    virtual_camera = meshpy.VirtualCamera(ci)
    scene_objs = {'object': meshpy.SceneObject(mesh, T_obj_world.inverse())}
    
    for name, scene_obj in scene_objs.iteritems():
        virtual_camera.add_to_scene(name, scene_obj)
        
    # camera pose
    cam_dist = 0.3
    T_camera_world = autolab_core.RigidTransform(rotation=np.array([[0, 1, 0],
                                                       [1, 0, 0],
                                                       [0, 0, -1]]),
                                        translation=[0,0,cam_dist],
                                        from_frame='camera',
                                        to_frame='world')
        
    T_obj_camera = T_camera_world.inverse() * T_obj_world

    renders = virtual_camera.wrapped_images(mesh, [T_obj_camera], RenderMode.COLOR,debug=False)



    vis.subplot(d,d,k+1)
    vis.imshow(renders[0].image)#.color)


rend_stop = time.time()

logging.info('Rendering complete, took %.3f sec' %(rend_stop-rend_start))
if not os.path.isdir('./faces'):

    os.mkdir('./faces')
vis.show("./faces/test.png")
if not os.path.isdir('./faces'):

    os.mkdir('./faces')
if not os.path.isdir('./faces'):

    os.mkdir('./faces')
#Output all views to subfolder
'''
for f, rgb in enumerate(rgb_l):

    img = Image.fromarray(rgb, 'RGB')
    img.save('faces/test_%i.png' % f)
    img.show()
'''
