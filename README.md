# semantic-map-object-recognition

This branch contains modules for different purposes, with the main ones
being:


## 1. Full pipeline (WIP)

Implemented in the Python script broader_pipeline.py 

### Stages

   * Supported Inputs: rosbag, npy matrix of image vectors, or path to image folder

   * OpenCV based Segmentation: currently reads the image frames as RGB and supports
     calc of histogram bins based on GMM Dirichlet process  (WIP)

   * Cluster merging is supported too, post GMM Dirichlet-based segmentation

   * The RGB image is also passed to a Python wrapper for a pre-trained version of YOLO
     to retrieve the bounding boxes for that image

     The YOLO bbox format is converted back to ROI form

   * Images are then cropped to the bounding box of maximum area


### Prerequisites

   * OpenCV
   * rosbag
   * lightnet
   * skimage (for RAG and cluster merging)

### TO-DOs

   * double check segmentation part based on archive branch

   * Add planar surface extraction and integration with point cloud input
     As started under ./pcl-plane-extraction/plane_extraction.py

   * Integrate with step 2 described below 


## 2. Image Matching against Shapenet (Classification only)

   Assumes that input images were already segmented, i.e., one object at a time   

   ### With Depth and color feature engineering

   Can be found in the shape_match.py script 

   ### With Feature descriptors in OpenCV

   Can be found in the feature_desc.py script

   ### With Siamese-like nets 

 
   The Keras implementation of NormXCorr by [1],
   also described in our paper [2],
   has been replaced and fully re-implemented in Pytorch.
   
   Currently we are testing a different, pre-trained architecture that combines
   two ResNet50 Siamese branches 
   with L2 embedding normalization similar to the fine-tuning routine in [3], 
   as illustrated in the docker branch.
   
   *Check docker branch for further details*.
   
   
   [1] Subramaniam, A., Chatterjee, M., & Mittal, A. (2016). Deep neural networks with inexact matching for person re-identification. 
   In Advances in Neural Information Processing Systems (pp. 2667-2675).
  
   [2] Chiatti, A., Bardaro, G., Bastianelli, E., Tiddi, I., Mitra, P., & Motta, E. (2019). 
   Exploring Task-agnostic, ShapeNet-based Object Recognition for Mobile Robots.
   
   [3] Qi, H., Brown, M., & Lowe, D. G. (2018). Low-shot learning with imprinted weights. 
   In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 5822-5830).
   
   ### Prerequisites

    * docker-ce
   


# Extras

## Matlab scripts to work with the NYUDepth dataset

   Under the NYUDepth-processing subfolder 
   To work with the 

   * count_objects.m can be reused to get classes of tagged objects and their cardinalities
   * maskfiles.m was used to mask objects belonging to a specific class out of each image and create
     a separate image for each frame (RGB object over Black background)
   
   ### Prerequisites
   
   * the NYUDepth  Dataset V2   https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html
   * the NYUDepth Matlab toolbox http://cs.nyu.edu/~silberman/code/toolbox_nyu_depth_v2.zip

## Histogram Visualizations

   Under the plot-histograms subfolder
   with GMM Dirichlet as above but also plotting resulting histograms

   

