# Code under development 


Dependencies / Requirements

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
  
