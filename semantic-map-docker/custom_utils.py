############################################
# This code is based on the Img2Vec project
# All credits go to here:
# https://github.com/christiansafka/img2vec
#
############################################

import os
import torch


def save_embeddings(model):


    #select the desired layer
    layer = model._modules.get('avgpool')

    embedding = torch.zeros(512) # i.e. output size of the last layer kept in ResNet

    #


