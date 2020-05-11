# Improving Object Recognition for Mobile Robots by Leveraging Common-sense Knowledge Bases 

In case you did not land here from [our main website](https://robots.kmi.open.ac.uk) already, be sure to check it out to see how 
these lines of code fit in our broader ideas! 

This repository contains all the code implemented for our latest experiments on few-shot object recognition for Mobile Robots, 
where we compare a baseline Nearest Neighbour approach to Image Matching,
with the K-net architecture described in [Zeng et al. (2018)](https://arxiv.org/pdf/1710.01330.pdf)  
and our own three novel versions of Image Matching pipeline:

- **Imprinted K-net** exploits a combination of K-net with weight imprinting, a techniques originally introduced on a standard CNN architecture by [Qi et al. (2018)](http://openaccess.thecvf.com/content_cvpr_2018/papers/Qi_Low-Shot_Learning_With_CVPR_2018_paper.pdf)

- **Imprinted K-net + ConceptRel** introduces common-sense concept relatedness from [ConceptNet Numberbatch](https://github.com/commonsense/conceptnet-numberbatch) at test time

- **Kground Imprinted K-net** was found to be the top-performing solution among our trials. Extended from the prior one, it also relies on spatial relationships between objects derived from [Visual Genome](https://visualgenome.org/) - through the link provided by [WordNet's](https://wordnet.princeton.edu/) synsets- for additional validation (see also Figure below). 

All the experimented NeuralNet-based architectures were implemented in Pytorch.

![framework-pic](https://github.com/kmi-robots/semantic-map-object-recognition/blob/master/assets/2019_framework_simpler.jpg "Kground Imprinted K-net")


We recommend to let Docker handle all the dependency installations, please refer to the dockerfile for a list of pre-requisites. 

## Pre-trained models & data

After cloning this repo, please download  
our pre-trained models and parsed KBs (e.g., Visual Genome) through
[this link](https://www.mediafire.com/file/zjpwnm17cbd5og6/starter_kit.zip/file).

All contents of starter_kit.zip will have to be extracted under ./semantic-map-docker/data :

```
cd path_to_download_folder
unzip starter_kit.zip -d your_path_to/semantic-map-object-recognition/object-recognition/data
```

## Getting Started

Here is a brief guide to our command-line interface to run/replicate the different configurations introduced above, once all dependencies have been
successfully installed. Please bear with us as we gradually Marie Kondo-nise this content!! :angel:


If not launching from Docker, after cloning this repo make sure to type:

```
cd semantic-map-docker
```

The `siamese_main.py` file is the one to be launched and expects different arguments, as listed below.
**By default**, if none of the optional arguments is modified and by running `python siamese_main.py json {train,test,baseline}
                       path_to_train path_to_test emb`, it will use the Kground Imprinted K-net pipeline on annotated bounding boxes, 
i.e., without any YOLO segmentation applied upfront.

```
usage: siamese_main.py [-h] [--resnet RESNET] [--sem {concept-only,full,None}]
                       [--transfer TRANSFER] [--store_emb STORE_EMB]
                       [--noimprint NOIMPRINT] [--twobranch TWOBRANCH]
                       [--plots PLOTS] [--model {knet, nnet,None}]
                       [--N {10,20,25}] [--K K]
                       [--Kvoting {majority,discounted}] [--batch BATCH]
                       [--lr LR] [--epochs EPOCHS] [--wdecay WDECAY]
                       [--patience PATIENCE] [--momentum MOMENTUM] [--avg AVG]
                       [--stn STN] [--mnist MNIST] [--ncc NCC] [--query QUERY]
                       {reference,pickled,json} {train,test,baseline}
                       path_to_train path_to_test emb

positional arguments:

  {reference,pickled,json}
                        Input type at test time: can be one between reference,
                        pickled or json (if environment data have been tagged)
  {train,test,baseline}
                        One between train, test or baseline (i.e., run
                        baseline NN at test time)
  path_to_train         path to training data in anchor branch, i.e., taken
                        from the environment. Assumes shapenet for product is
                        available in the local branch
  path_to_test          path to test data, in one of the three formats listed
                        above
  emb                   path where to store/retrieve the output embeddings

optional arguments:

  -h, --help            show this help message and exit
  --resnet RESNET       makes ResNet the building block of the CNN, True by
                        default
  --sem {concept-only,full,None}
                        whether to add the semantic modules at inference time.
                        It includes both ConceptNet and Visual Genome by
                        defaultSet to concept-only to discard Visual Genome or
                        None to test without any semantics
  --transfer TRANSFER   Defaults to False in all our reported trials. Changeto
                        True if performing only transfer learning without
                        fine-tuning
  --store_emb STORE_EMB
                        Defaults to True. Set to false if do not wish to
                        storethe training embedddings locally
  --noimprint NOIMPRINT
                        Defaults to False. Set to through if running without
                        weight imprinting
  --twobranch TWOBRANCH
                        Defaults to False in all our reported trials. Can be
                        set to True to test a Siamese with weights learned
                        independently on each branch
  --plots PLOTS         Optionally produces plot to check and val loss
  --model {knet, nnet,None}
                        set to pick one between K-net or N-net or None. K-net
                        is used by default
  --N {10,20,25}        Number of object classes. Should be one between
                        10,20,25. Defaults to 25
  --K K                 Number of neighbours to consider for ranking at test
                        time (KNN). Defaults to 5
  --Kvoting {majority,discounted}
                        How votes are computed for K>1. Defaults to discounted
                        majority voting.
  --batch BATCH         Batch size. Defaults to 16
  --lr LR               Learning rate. Defaults to 0.0001
  --epochs EPOCHS       Number of training epochs. Defaults to 5000
  --wdecay WDECAY       Weight decay. Defaults to 0.00001
  --patience PATIENCE   Number of subsequent epochs of no improvement before
                        early stop. Defaults to 100
  --momentum MOMENTUM   momentum for SGD training. Defaults to 0.9
  --avg AVG             metric average type for training monitoring and plots.
                        Please refer to the sklearn docs for all allowed
                        values.
  --stn STN             Defaults to False in all our reported trials. Can be
                        set to True to add a Spatial Transformer Net before
                        the ResNet module
  --mnist MNIST         whether to test on the MNIST benchmark dataset
  --ncc NCC             whether to test with the NormXCorr architecture
  --query QUERY         Path to data used with support on test time in prior
                        trials
```

Please note that automatic re-sizing for any number of classes is currently not supported, we currently support only N=10,15 and 25 (See also the [Data set Section](#data-sets))




## Sample command combinations

Commands to replicate pipelines under evaluation in the latest experiments.

For running the Kground imprinted K-net just type:

```
python siamese_main.py json {train,test} path_to_train path_to_test emb

```

For the Imprinted K-net + ConceptRel:

```

python siamese_main.py json {train,test} path_to_train path_to_test emb --sem concept-only
```

For the Imprintent K-net without semantics

```

python siamese_main.py json {train,test} path_to_train path_to_test emb --sem None

```

For our own implementation of K-net from the paper by [Zeng et al. (2018)](https://arxiv.org/pdf/1710.01330.pdf):

```
python siamese_main.py json {train,test} path_to_train path_to_test emb --noimprint True --sem None --K 1 

```

For a baseline nearest neighibour match at test time:


```
python siamese_main.py {reference, pickled, json} baseline path_to_train path_to_test emb 
```


## Data sets

Some released data are available under `semantic-map-docker/data`. The relevant sets documented here are:

*  the **ShapeNet + Google (SNG) sets** are saved under `shapenet10`, `shapenet20` and `shapenet25` respectively, depending on the number of object classes examined.
*  `KMi_collection` is for the reference images collected in KMi, but excludes the test image set.
*  ` yolo` contains all files for using the pre-trained YOLO version embedded in OpenCV. These files are referenced by the `segment.py` methods 


## Code map

If you are interested in further customising this code, here are more details on the content of this branch. Some of the files are included for completeness, but <s>striked out</s>, to indicate they are not needed for reproducing the experiments described above. Please bear with us as we gradually Marie Kondo-nise this content!! :angel:

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

    - `common_sense.py` groups all methods/heuristics used to propose and validate corrections based on external common-sense knowledge. 
    
    -  `plot_results.py` generates plots of the type "Metric vs Epoch" in Visdom (i.e., also available from browser when launched on remote server). Metrics currently implemented:
   loss, accuracy, precision, and recall. 

    - `pytorchtools.py`, a script implemented at [Bjarten/early-stopping-pytorch](https://github.com/Bjarten/early-stopping-pytorch) to resemble the handy Keras callbacks. Here used to set the early stopping criterion on validation.

    - <s>rpooling.py</s> and <s>cirtorch_functional.py</s> refer to trials on region proposal layers that are incomplete or were not incorporated.
 
    - <s>ncc-extension</s> includes all the C++ code reproducing the NormXCorr custom layer described in [Submariam et al. (2016)](http://papers.nips.cc/paper/6367-deep-neural-networks-with-inexact-matching-for-person-re-identification.pdf) to speed up CPU computation. It can be manually installed by running `python setup.py install` within this folder.
The GPU version, i.e., using C++ for CUDA parallel programming (`ncc_cuda.cpp`, `ncc_cuda.cu` and `setup_cuda.py`) is still incomplete. 

    - <s>data_preparation.py</s> was an older script for data preparation 


* `utils` contains some handy scripts used mainly for data parsing from/to different formats. More specifically:

    - `VG_relations.py` was used to extract a subset of spatial relationships for the experiments introduced above.

    - `bag_converter.py` is a Python 2.6 script to read images directly from the ROSbag format and save them locally in pickle format. Optionally, sequences of images can
      be temporally sampled, based on the default timestamps available from the ROSbag.

    - <s> ImageNet_attrinit.py </s>  was started to include ImageNet attributes but was abandoned, in the end, in favour of Visual Genome, which already includes those attributes
    - <s> base_DA.py </s>  is a static alternative to data augmentation run on data loading in pytorch, also out of scope of the above experiments.


* For more details on the content of the `data` folder please refer to the [Data sets]("#data-sets") Section.   


