######################################################################
# This script is a minor modification of code from the Img2Vec project
# All credits go to the contributors at:
# https://github.com/christiansafka/img2vec
#
######################################################################

import torch


def save_embeddings(model, path_to_state, path_to_data,  device):

    #select the desired layer as of latest checkpoint
    model = model._modules.get('embed')
    model.load_state_dict(torch.load(path_to_state))

    model.eval()

    #i.e., avgpool layer of embedding Net
    layer = model._modules.get('resnet')._modules.get('avgpool')

    data, labels, ids = torch.load(path_to_data)

    embeddings = {}

    for img, target, obj_id in data:

        img = img.to(device)

        embedding = torch.zeros(512).to(device) # i.e. output size of the last layer kept in ResNet

        #define a function to copy the output of a layer
        def copy_data(m,i,o):

            embedding.copy_(o.data.squeeze())

        #Attach the function to a selected layer
        h = layer.register_forward_hook(copy_data)

        #Run model on each image
        model.get_embeddings(img)

        #Detach copy function from the layer
        h.remove()

        #Save embedding in dictionary under unique id
        embeddings[obj_id] = embedding

    return embeddings
