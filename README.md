# semantic-map-object-recognition

This branch contains modules for different purposes, with the main ones
being:


## 1. Full pipeline (WIP)

broader_pipeline.py

### Stages

   * Supported Inputs: rosbag, npy matrix of image vectors, or path to image folder

   * OpenCV based Segmentation  (WIP)

   *  Bounding boxes from YOLO

   * Cropping to main bbox

   * Clustering by GMM Dirichlet

   * Post merge of clusters 


### Prerequisites

   * OpenCV
   * rosbag
   * lightnet

###TO-DOs

   * Add planar surface extraction and integration with point cloud input
     As started under ./pcl-plane-extraction/plane_extraction.py

   * Integrate with step 2 described below 


## 2. Image Matching against Shapenet (Classification only)

   Assumes input images to be already segment to include one object at a time   

   with depth and color feature engineering

   with feature descriptors in OpenCV

   with Siamese nets 
 
   (via Docker)




# Extras

## Matlab scripts to work with the NYUDepth dataset


## Histogram Visualizations


   with GMM Dirichlet

