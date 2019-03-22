"""
This script is to save
embeddings from the last conv layer
of a ResNet, after L2-norm,
inspired by paper by Qi et al. (CVPR 2018)

"""

import torch


#target_dim = 256 #2048 #512  # i.e. output size of the last layer kept in ResNet


def save_embeddings(model, path_to_state, path_to_data, device, transforms=None):

    model.load_state_dict(torch.load(path_to_state))

    model.eval()

    data = torch.load(path_to_data)

    embeddings = {}

    for img_id, img in data.items():

        #Applying same normalization as on a training forward pass
        img[0,:] = transforms(img[0,:].float())

        img = img.float().to(device)

        #Save embedding in dictionary under unique id
        embeddings[img_id] = model.get_embedding(img) #embedding


    #Save dictionary locally, as JSON file
    with open('./out_embeddings.dat', mode='wb') as outf:
        torch.save(obj=embeddings, f=outf)

