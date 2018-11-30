# semantic-map-object-recognition

This branch contains older scripts with different purposes, based on some exploratory attempts.

## 1. Working with the Visual Genome Dataset

visual_genome_filter.py is a script for pre-processing Visual Genome according to candidate keywords (saved in form of JSON dictionary under ./data)

It was implemented initially to filter out as many outdoor examples as possible from this data set and focus on indoor scenes.

### Pre-requisites

 * visual_genome_python_driver
 * Download (https://visualgenome.org/api/v0/api_home.html) scene_graphs.json, synsets.json, image_data.json
   under data/
 * concepts_larger.json is a dictionary of candidate keywords to filter Visual Genome and select indoor scenes, i.e., a   a combination of ConceptNet "AtLocation" objects and our own defined frames
 * NLTK for lemmatization 


## 2. Segmentation based on Open CV and ROS bags

Trial on segmenting based on depth-only related features.
Discontinued at the moment.

### Pre-requisites

  * OpenCV
  * rosbag package


## 3.1 Parsing 3D ShapeNet models into 2D views

The 3D-to-2D subfolder contains a script that creates a faceted viz of different 2D planes identified from a 
ShapeNet 3D object of extension .obj

### Pre-requisites 

  * ShapeNet (https://www.shapenet.org)
  * meshpy (https://documen.tician.de/meshpy) 


## 4. Skeleton detection trials

To be fixed and discontinued at the moment.
