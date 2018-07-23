import os
import sys
from matplotlib import pyplot as plt
import meshpy
import autolab_core
from perception import CameraIntrinsics
from PIL import Image
import numpy as np 

#TODO:
# scale to more objs
# keep ids and output in sub-folders grouped by id

#Read 3D model
mesh = meshpy.ObjFile('model_normalized.obj').read()


########CAMERA SETUP#################################################################
#intrinsic matrix values are derived from K matrix in ros message
#projection matrix is indicated as P in ros
# (ros has a 3x4 default for P so only the left 3x3 portion has to be considered)

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

#Defining obj to camera poses
#z axis away from scene, x to right, y up
rt = autolab_core.RigidTransform(

    rotation= np.array([[0., 0., -1.], [0., 1., 0.],[1.,0., 0.]]),
    from_frame='camera',
    to_frame='obj'
)

cam = meshpy.VirtualCamera(ci)

rgb_l, depth_l = cam.images(mesh, [rt])

#Output all views to subfolder
if not os.path.isdir('./faces'):

    os.mkdir('./faces')

for f, rgb in enumerate(rgb_l):

    img = Image.fromarray(rgb, 'RGB')
    img.save('faces/test_%i.png' % f)
    img.show()

