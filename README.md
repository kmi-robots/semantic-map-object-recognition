# semantic-map-object-recognition

This branch contains only files for the docker implementations:

* the `dockerfile` creates a docker image for the environment all needed dependencies 
used to run our latest experiments, where we compare a baseline Nearest Neighbour approach to Image Matching,
with the K-Net architecture described in [Zeng et al. (2018)](https://arxiv.org/pdf/1710.01330.pdf)  
and our own three novel versions of Image Matching pipeline:

- one exploiting a combination of K-Net with weight imprinting, a techniques originally introduced on a standard CNN architecture by [Qi et al. (2018)](http://openaccess.thecvf.com/content_cvpr_2018/papers/Qi_Low-Shot_Learning_With_CVPR_2018_paper.pdf)

- one introducing common-sense concept relatedness from [ConceptNet Numberbatch](https://github.com/commonsense/conceptnet-numberbatch) at test time

- the top-performing solution, extended from the prior one, also relying on spatial relationships between objects derived from [Visual Genome](https://visualgenome.org/) - through the link provided by [WordNet's](https://wordnet.princeton.edu/) synsets- for additional validation. 

All the experimented NeuralNet-based architectures were implemented in Pytorch.

The dockerfile extends the [Deepo docker image for deep learning](https://hub.docker.com/r/ufoym/deepo/) for Python3, and already handles the installation of all dependecies 
(*scikit-learn*, *torch*, *torchvision*, *visdom*)) and also installs our own C++ extension to compute Normalized Cross Correlation in Pytorch that was implemented after running the early trials described [here](oro.open.ac.uk/59508/1/DARLI_AP_2019_Paper.pdf).


* `semantic-map-docker`, groups all the code needed to run the different models under comparison. 

    - The models architectures are defined in the `siamese_models.py` file

    - `data_loaders.py` was used to load: the MNIST data, extended to form triplets in the format expected by the Siamese models;
 a custom dataset derived from Shapenet, i.e., the same method can be easily modified to load any custom data locally available.

    -  `plot_results.py` generates plots of the type "Metric vs Epoch" in Visdom (i.e., also available from browser when launched on remote server). Metrics currently implemented:
   loss, accuracy, precision, and recall. 

    - `pytorchtools.py`, a script implemented at [Bjarten/early-stopping-pytorch](https://github.com/Bjarten/early-stopping-pytorch) to resemble the handy Keras callbacks. Here used to set the early stopping criterion on validation.

    - This tool can also be run locally after cloning this repo and manually installing all dependencies, through `python siamese_main.py` (Python3)
 
    - `ncc-extension` includes all the C++ code reproducing the NormXCorr custom layer described in [Submariam et al. (2016)](http://papers.nips.cc/paper/6367-deep-neural-networks-with-inexact-matching-for-person-re-identification.pdf) to speed up CPU computation. It can be manually installed by running `python setup.py install` within this folder.
The GPU version, i.e., using C++ for CUDA parallel programming (`ncc_cuda.cpp`, `ncc_cuda.cu` and `setup_cuda.py`) is still incomplete. 
