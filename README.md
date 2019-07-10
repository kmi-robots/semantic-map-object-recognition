# Improving Object Recognition for Mobile Robots by Leveraging Common-sense Knowledge Bases 

In case you did not land here from [our main website](https://robots.kmi.open.ac.uk) already, be sure to check it out to see how 
all these lines of code fit in our broader ideas! 

This repository contains all the code implemented for our latest experiments on few-shot object recognition for Mobile Robots, 
where we compare a baseline Nearest Neighbour approach to Image Matching,
with the K-net architecture described in [Zeng et al. (2018)](https://arxiv.org/pdf/1710.01330.pdf)  
and our own three novel versions of Image Matching pipeline:

- *Imprinted K-net* exploits a combination of K-net with weight imprinting, a techniques originally introduced on a standard CNN architecture by [Qi et al. (2018)](http://openaccess.thecvf.com/content_cvpr_2018/papers/Qi_Low-Shot_Learning_With_CVPR_2018_paper.pdf)

- *Imprinted K-net + ConceptRel* introduces common-sense concept relatedness from [ConceptNet Numberbatch](https://github.com/commonsense/conceptnet-numberbatch) at test time

- *Kground Imprinted K-net* was found to be the top-performing solution among our trials. Extended from the prior one, it also relies on spatial relationships between objects derived from [Visual Genome](https://visualgenome.org/) - through the link provided by [WordNet's](https://wordnet.princeton.edu/) synsets- for additional validation (see also Figure below). 

All the experimented NeuralNet-based architectures were implemented in Pytorch.

![framework-pic](https://github.com/kmi-robots/semantic-map-object-recognition/blob/master/assets/2019_framework_simpler.jpg "Kground Imprinted K-net")


We recommend to let Docker handle all the dependency installations, please refer to the dockerfile for a list of pre-requisites. 

## Getting Started

Here is a brief guide to our command-line interface to run/replicate the different configurations introduced above, once all dependencies have been
successfully installed. 

The `siamese_main.py` file is the one to be launched and expects different parameters:

```

```


## Data sets
WORK IN PROGRESS

## Code map

If you are interested in further customising this code, here are more details on the content of this branch. Some of the files are included for completeness, but <s>striked out</s>, to indicate they are not needed for reproducing the experiments described above. Please bear with us as we gradually Marie Kondo-nise this content!! 

* the `dockerfile` creates a docker image for the environment all needed dependencies 
used to run our latest experiments,The dockerfile extends the [Deepo docker image for deep learning](https://hub.docker.com/r/ufoym/deepo/) for Python3, and already handles the installation of all dependecies 
(*scikit-learn*, *torch*, *torchvision*, *visdom*)) and also installs our own C++ extension to compute Normalized Cross Correlation in Pytorch that was implemented after running the early trials described [here](oro.open.ac.uk/59508/1/DARLI_AP_2019_Paper.pdf).

* the `semantic-map-docker` folder, grouping all the code needed to run the different models under comparison. 

    - The models architectures are defined in the `siamese_models.py` file

    - The main tool lifecycle called from the `siamese_main.py` follows a standard PyTorch structured and is organised under the `train.py`, `validate.py` and       `test.py` files

    - `segment.py` ensures the integration with YOLO in OpenCV, when bounding box/object detection is also activated. It was extended from code shown in this [tutorial](https://github.com/meenavyas/Misc/blob/master/ObjectDetectionUsingYolo/ObjectDetectionUsingYolo.ipynb), as it also combines saliency region detection to potentially increase the number of bounding boxes found. 

    - `imprint.py` contains the routines called only when weight imprinting is activated. For further details on how this works, you can refer to the [original paper on weight imprinting](http://openaccess.thecvf.com/content_cvpr_2018/papers/Qi_Low-Shot_Learning_With_CVPR_2018_paper.pdf) or have a look to [this repository](https://github.com/YU1ut/imprinted-weights), which provides a non-official Pytorch implementation for ResNet-based architectures. 

    - `baseline_KNN.py`, as the name suggests, is the code portion that handles baseline nearest neighbour, whenever the run setup requires to run the baseline.
 
    - `data_loaders.py` includes all custom classes implemented to load the input images as balanced 1:1 triplets in the form anchor, positive example, negative example. 

    - `embedding_extractor.py` is used to extract embeddings from the pre-trained architectures used and return the embedding space needed to classify objects at test time. Different methods indicate that either a different model or a different format for the input image was utilised on extraction. 

    -  `plot_results.py` generates plots of the type "Metric vs Epoch" in Visdom (i.e., also available from browser when launched on remote server). Metrics currently implemented:
   loss, accuracy, precision, and recall. 

    - `pytorchtools.py`, a script implemented at [Bjarten/early-stopping-pytorch](https://github.com/Bjarten/early-stopping-pytorch) to resemble the handy Keras callbacks. Here used to set the early stopping criterion on validation.

    - <s>rpooling.py</s> and <s>cirtorch_functional.py</s> refer to trials on region proposal layers that are incomplete or were not incorporated.
 
    - <s>ncc-extension</s> includes all the C++ code reproducing the NormXCorr custom layer described in [Submariam et al. (2016)](http://papers.nips.cc/paper/6367-deep-neural-networks-with-inexact-matching-for-person-re-identification.pdf) to speed up CPU computation. It can be manually installed by running `python setup.py install` within this folder.
The GPU version, i.e., using C++ for CUDA parallel programming (`ncc_cuda.cpp`, `ncc_cuda.cu` and `setup_cuda.py`) is still incomplete. 


* `utils` contains some handy scripts used mainly for data parsing from/to different formats. More specifically:

    -

* For more details on the content of the `data` folder please refer to the "Data sets" Section.   


