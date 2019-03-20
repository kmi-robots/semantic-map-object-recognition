"""
This script groups all modules used to extract,
process (e.g., via RMAC pooling) and save
embeddings from the second last layer
of a ResNet

"""

import torch
import cirtorch_functional as CF

target_dim = 2048 #512  # i.e. output size of the last layer kept in ResNet


def save_embeddings(model, path_to_state, path_to_data, device, transforms=None):

    """
    This function was derived from code from the Img2Vec project:
    https://github.com/christiansafka/img2vec
    """


    #select the desired layer as of latest checkpoint
    model = model._modules.get('embed').to(device)
    model.load_state_dict(torch.load(path_to_state))

    model.eval()

    #i.e., avgpool layer of embedding Net
    layer = model._modules.get('resnet')._modules.get('avgpool').to(device)

    data = torch.load(path_to_data)

    #embeddings = torch.Tensor((data.shape[0], 1, target_dim)).to(device)
    embeddings = {}

    for img_id, img in data.items():

        #Applying same normalization as on a training forward pass
        img[0,:] = transforms(img[0,:].float())

        img = img.float().to(device)

        embedding = torch.zeros(target_dim).to(device)

        #define a function to copy the output of a layer
        def copy_data(m,i,o):

            embedding.copy_(o.data.squeeze())

        #Attach the function to a selected layer
        h = layer.register_forward_hook(copy_data)

        #Run model on each image
        model.get_embedding(img) #It is only one forward pass on one single pipeline of the original siamese

        #Applying RMAC pooling to the extracted embedding
        print(embedding.shape)
        embedding = CF.rmac(embedding)
        print(embedding.shape)

        #Detach copy function from the layer
        h.remove()

        #Save embedding in dictionary under unique id
        embeddings[img_id] = embedding


    #Save dictionary locally, as JSON file
    with open('./out_embeddings.dat', mode='wb') as outf:
        torch.save(obj=embeddings, f=outf)

