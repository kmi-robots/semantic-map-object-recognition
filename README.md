# Code under development 


## Dependencies / Requirements

This code has been tested for / requires 

- Python 3.5 or later
- ROS kinetic / ROS melodic  + rospkg and catkin-pkg
- We also recommend to add the following line to your .bashrc

```
    export ROS_PYTHON_VERSION=3
```

- It requires that OpenCV bridge and TF link I/O with ROS is built for Python3 (by default it works only for Python2)
  [This answer](https://stackoverflow.com/questions/49221565/unable-to-use-cv-bridge-with-ros-kinetic-and-python3) can be taken as reference for CV bridge
  [a similar approach](https://github.com/ros/geometry2/issues/259) can be used to import tf in Python3

- Pytorch +  Torchvision (CPU-only is enough to run the inference without retraining, for the training we tested the pipeline on CUDA 9) 

- opencv, opencv-contrib, mlxtend, scikit-learn(v.0.21 recommended for compatibility with mlxtend), matplotlib

- numpy 1.14 or later. Make sure to remove links created to the python2 version of numpy by ROS or to keep only one version installed 
  after upgrading.

- the pattern and requests packages 


- nltk and the Wordnet corpus specifically. We found that installing through pip was not successful so we suggest the following steps:

```
  sudo apt-get install python3-nltk

  python3

  >> import nltk
  >> nltk.download('wordnet')

```

- the Pythia VQA model developed at Facebook is also used for inference.
  Please refer to their [official tutorial](https://colab.research.google.com/drive/1Z9fsh10rFtgWe4uy8nvU4mQmqdokdIRR) for the relevant installation steps and dependencies
  
  
## Getting started

### Pre-trained models & data

After cloning this repo, please download  our pre-trained models and parsed KBs (e.g., Visual Genome) through
[this link](https://www.mediafire.com/file/zjpwnm17cbd5og6/starter_kit.zip/file).

All contents of starter_kit.zip will have to be extracted under ./object-recognition/data :

```
cd path_to_download_folder
unzip starter_kit.zip -d your_path_to/semantic-map-object-recognition/object-recognition/data
```


### Test commands

The inference pipeline can be tested with, e.g.:

```
python3 main.py camera explore ./data/KMi_collection/train ./data/KMi_collection/test/tagged_KMi.json ./pt_results/kmish25/embeddings_imprKNET_1prod.dat --bboxes segmented --sem none
```

In this case the scene will be also segmented (--boxes segmented) and no common-sense based correction
will be applied (--sem none). You can run ```python3 main.py --help``` for more options:


This command will launch the pipeline and be in the lookout for a service call
to start the exploration (i.e., object recognition/image analysis). 
You can send this call through the following command for testing/debugging

```
rosservice call /start_exploration "data: true"
```
or define a ROS service sending the same command when, e.g., the robot 
has reached a specific waypoint (an sample script which uses this rationale 
is available in our [dh_interaction repo](https://github.com/kmi-robots/dh_interaction/blob/master/scripts/navigation_exploration.py))

 
