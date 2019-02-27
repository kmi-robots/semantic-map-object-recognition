# semantic-map-object-recognition

This branch contains only files for the docker implementations:

* the `dockerfile` used to build the images already
available on the Docker Hub and implementing the architecture 
of [Submariam et al. (2016)](http://papers.nips.cc/paper/6367-deep-neural-networks-with-inexact-matching-for-person-re-identification.pdf), 
as well as a simpler siamese-like architecture using absolute distance as similarity measure.
All code is implemented in Pytorch.
The dockerfile extends the [Deepo docker image for deep learning](https://hub.docker.com/r/ufoym/deepo/) for Python3, and already handles the installation of all dependecies 
(*scikit-learn*, *torch*, *torchvision*, *visdom*)) and also installs our own C++ extension to compute Normalized Cross Correlation in Pytorch.


* `semantic-map-docker`, groups all the code needed to run two versions of siamese-like models, described in `siamese_models.py`

* `data_loaders.py` was used to load: the MNIST data, extended to form triplets in the format expected by the Siamese models;
 a custom dataset derived from Shapenet, i.e., the same method can be easily modified to load any custom data locally available.

* `plot_results.py` generates plots of the type "Metric vs Epoch" in Visdom (i.e., also available from browser when launched on remote server). Metrics currently implemented:
   loss, accuracy, precision, and recall. 

* `pytorchtools.py`, a script implemented at [Bjarten/early-stopping-pytorch](https://github.com/Bjarten/early-stopping-pytorch) to resemble the handy Keras callbacks. Here used to set the early stopping criterion on validation.

* This tool can also be run locally after cloning this repo and manually installing all dependencies, through `python siamese_main.py` (Python3)

* `ncc-extension` includes all the C++ code reproducing the NormXCorr custom layer described in [Submariam et al. (2016)](http://papers.nips.cc/paper/6367-deep-neural-networks-with-inexact-matching-for-person-re-identification.pdf) to speed up CPU computation. It can be manually installed by running `python setup.py install` within this folder.
The GPU version, i.e., using C++ for CUDA parallel programming (`ncc_cuda.cpp`, `ncc_cuda.cu` and `setup_cuda.py`) is still incomplete. 
