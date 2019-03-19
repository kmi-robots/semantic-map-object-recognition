######################################################################
# This script is a minor modification of code from the Img2Vec project
# All credits go to the contributors at:
# https://github.com/christiansafka/img2vec
#
######################################################################

import torch
import json
target_dim = 512  # i.e. output size of the last layer kept in ResNet

def save_embeddings(model, path_to_state, path_to_data,  device):

    #select the desired layer as of latest checkpoint
    model = model._modules.get('embed')
    model.load_state_dict(torch.load(path_to_state))

    model.eval()


    #i.e., avgpool layer of embedding Net
    layer = model._modules.get('resnet')._modules.get('avgpool')

    data, labels, ids = torch.load(path_to_data)

    #embeddings = torch.Tensor((data.shape[0], 1, target_dim)).to(device)
    embeddings = {}

    for img, target, obj_id in data:

        img = img.to(device)

        embedding = torch.zeros(target_dim).to(device)

        #define a function to copy the output of a layer
        def copy_data(m,i,o):

            embedding.copy_(o.data.squeeze())

        #Attach the function to a selected layer
        h = layer.register_forward_hook(copy_data)

        #Run model on each image
        model.get_embeddings(img) #It is only one forward pass on one single pipeline of the original siamese

        #Detach copy function from the layer
        h.remove()

        #Save embedding in dictionary under unique id
        embeddings[obj_id] = embedding
        #embeddings[index, :] =


    #Save dictionary locally, as JSON file
    with open('./out_embeddings.json', encoding='utf-8', mode='w') as outjson:
        json.dump(embeddings, outjson)

