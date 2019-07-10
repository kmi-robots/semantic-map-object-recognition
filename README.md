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

![framework-pic]( "Kground Imprinted K-net")


We recommend to let Docker handle all the dependency installations, please refer to the dockerfile for a list of pre-requisites. 

## Getting Started

Here is a brief guide to our command-line interface to run/replicate the different configurations introduced above, once all dependencies have been
successfully installed. 
WORK IN PROGRESS

## Code map

If you are interested in further customising this code, here are more details on the content of this branch.

* the `dockerfile` creates a docker image for the environment all needed dependencies 
used to run our latest experiments,The dockerfile extends the [Deepo docker image for deep learning](https://hub.docker.com/r/ufoym/deepo/) for Python3, and already handles the installation of all dependecies 
(*scikit-learn*, *torch*, *torchvision*, *visdom*)) and also installs our own C++ extension to compute Normalized Cross Correlation in Pytorch that was implemented after running the early trials described [here](oro.open.ac.uk/59508/1/DARLI_AP_2019_Paper.pdf).

* the `semantic-map-docker` folder, grouping all the code needed to run the different models under comparison. 

    - The models architectures are defined in the `siamese_models.py` file

    - `data_loaders.py` was used to load: the MNIST data, extended to form triplets in the format expected by the Siamese models;
 a custom dataset derived from Shapenet, i.e., the same method can be easily modified to load any custom data locally available.

    -  `plot_results.py` generates plots of the type "Metric vs Epoch" in Visdom (i.e., also available from browser when launched on remote server). Metrics currently implemented:
   loss, accuracy, precision, and recall. 

    - `pytorchtools.py`, a script implemented at [Bjarten/early-stopping-pytorch](https://github.com/Bjarten/early-stopping-pytorch) to resemble the handy Keras callbacks. Here used to set the early stopping criterion on validation.

    - This tool can also be run locally after cloning this repo and manually installing all dependencies, through `python siamese_main.py` (Python3)
 
    - `ncc-extension` includes all the C++ code reproducing the NormXCorr custom layer described in [Submariam et al. (2016)](http://papers.nips.cc/paper/6367-deep-neural-networks-with-inexact-matching-for-person-re-identification.pdf) to speed up CPU computation. It can be manually installed by running `python setup.py install` within this folder.
The GPU version, i.e., using C++ for CUDA parallel programming (`ncc_cuda.cpp`, `ncc_cuda.cu` and `setup_cuda.py`) is still incomplete. 
