import pcl

cloud = pcl.load('./point_clouds/camera_only.pcd')

print(cloud.size)

fil = cloud.make_passthrough_filter()
fil.set_filter_field_name("z")
fil.set_filter_limits(0, 1.5)
cloud_filtered = fil.filter()

print(cloud_filtered.size)

seg = cloud_filtered.make_segmenter_normals(ksearch=50)
seg.set_optimize_coefficients(True)
seg.set_model_type(pcl.SACMODEL_PLANE)
seg.set_normal_distance_weight(0.1)
seg.set_method_type(pcl.SAC_RANSAC)
seg.set_max_iterations(100)
seg.set_distance_threshold(0.03)
indices, model = seg.segment()

print(model)

cloud_plane = cloud_filtered.extract(indices, negative=False)
# NG : const char* not str
# cloud_plane.to_file('table_scene_mug_stereo_textured_plane.pcd')
pcl.save(cloud_plane, './camera_only_plane.pcd')

