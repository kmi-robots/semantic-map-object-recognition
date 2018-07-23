import os
import sys
#import pywavefront
#import pymesh
from matplotlib import pyplot as plt
import meshpy
import autolab_core
from perception import CameraIntrinsics
from PIL import Image

'''
scene = pywavefront.Wavefront('model_normalized.obj')


for name, material in scene.materials.items():
    print(name)
    print(material.vertices)

'''
'''
#Read full obj file
with open('model_normalized.obj') as model:

    lines = model.readlines()

#List of vertices 
vert_raw = [ line.split('\n')[0] for line in lines if line[0] =='v' and line[1]!='n']

print(vert_raw)
print(len(vert_raw))
'''
mesh = meshpy.ObjFile('model_normalized.obj').read()
#mesh = pymesh.load_mesh('model_normalized.obj')
#print(mesh)

rt = autolab_core.RigidTransform()

#Camera Intrinsics - based on Kinect 
ci = CameraIntrinsics(

    frame='camera',
    fx = 594.0502248744873,
    fy = 597.6261068216613,
    cx = 324.761877545329,
    cy = 234.3706583714745,
    skew=0.0,
    height=480,
    width=640
)


cam = meshpy.VirtualCamera(ci)

rgb_l, depth_l = cam.images(mesh, [rt])

if not os.path.isdir('./faces'):

    os.mkdir('./faces')

for f, rgb in enumerate(rgb_l):

    img = Image.fromarray(rgb, 'RGB')
    img.save('faces/test_%i.png' % f)
    img.show()

