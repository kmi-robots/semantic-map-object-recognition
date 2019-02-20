# semantic-map-object-recognition

This branch contains only files for the docker implementations:

* the *dockerfile* used to build the images already
available on the Docker Hub and implementing the architecture 
of [Submariam et al. (2016)](http://papers.nips.cc/paper/6367-deep-neural-networks-with-inexact-matching-for-person-re-identification.pdf) 
in Keras, with Tensorflow backend. 

* *data_preparation.py*, a Python script that can be used to generate positive and negative examples
   to work with siamese-like networks, from sets of input images (organized in folders by object class).
   Output data and label sets are saved to local disk in .npy format. 



