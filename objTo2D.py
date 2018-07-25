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
import argparse


#TODO:
#separate in different images per view, now faceted

if __name__ == '__main__':

    logging.getLogger().setLevel(logging.INFO)

    #logging.getLogger().setLevel(logging.ERROR)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('datadir', help='path to root dir for shapenet core dataset')

    args = parser.parse_args()

    #Hardcoded: list of synsets of interest
    #from taxonomy.json
    syns = ['03001627','03797390', '03085013', '03211117','03991062','04256520','04004475','03642806','03636649', '03337140']     #'02871439']
    out_name= './2D-views'

    if not os.path.isdir(out_name):

        os.mkdir(out_name)

    synpaths = [os.path.join(args.datadir, syn) for syn in syns]

    #print(synpaths)    

    fullp=[]
    for path in synpaths: 
    
        levtwo = os.listdir(path)
        
        fullp.extend([os.path.join(path, p) for p in levtwo])
    
    #print(fullp)
    #sys.exit(0)
    for pt in fullp:

        read_start= time.time()

        comps = pt.split('/')
        f_id = comps[len(comps)-1]
        synset = comps[len(comps)-2]       
 
        if not os.path.isdir(os.path.join(out_name,synset)):
            os.mkdir(os.path.join(out_name,synset))
        
        if not os.path.isdir(os.path.join(out_name,synset,f_id)):
            os.mkdir(os.path.join(out_name,synset,f_id))

        try:
            mesh = meshpy.ObjFile(os.path.join(pt,'models/model_normalized.obj')).read()
        
        except Exception as e:

            print(str(e))   
            #logging.error(str(e)) 
            
            continue       

        read_stop = time.time()
        logging.info('Read took %.3f sec' %(read_stop-read_start))

        sub_start= time.time()
        mesh = mesh.subdivide() #min_tri_length=0.01)
        mesh.compute_vertex_normals()
        stable_poses = mesh.stable_poses()
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


            mat_props = meshpy.MaterialProperties(color=(0,255,0),
                                         ambient=0.5,
                                         diffuse=1.0,
                                         specular=1,
                                         shininess=0)
            # create material props
            '''
            mat_props = meshpy.MaterialProperties(color=(255,255,255), 
                                    ambient=-1.0,
                                    diffuse=-1.0,
                                    shininess=-1.0
                                     )
            '''
            scene_objs = {'object': meshpy.SceneObject(mesh, T_obj_world.inverse())}
    
            for name, scene_obj in scene_objs.iteritems():
                virtual_camera.add_to_scene(name, scene_obj)
        
            # camera pose
            cam_dist = 2.0
            T_camera_world = autolab_core.RigidTransform(rotation=np.array([[0, 1, 0],
                                                       [1, 0, 0],
                                                       [0, 0, -1]]),
                                        translation=[0,0,cam_dist],
                                        from_frame='camera',
                                        to_frame='world')
        
            T_obj_camera = T_camera_world.inverse() * T_obj_world

            renders = virtual_camera.wrapped_images(mesh, [T_obj_camera], RenderMode.COLOR, mat_props= mat_props, debug=False)



            vis.subplot(d,d,k+1)
            vis.imshow(renders[0].image)#.color)
            

        vis.show(os.path.join(out_name,synset,f_id,'views.png'))
        vis.clf()
        #sys.exit(0)

        rend_stop = time.time()

        logging.info('Rendering complete, took %.3f sec' %(rend_stop-rend_start))
    

        #vis.show("./faces/test.png")

        #Output all views to subfolder
        '''
        for f, rgb in enumerate(rgb_l):

            img = Image.fromarray(rgb, 'RGB')
            img.save('faces/test_%i.png' % f)
            img.show()
        '''
