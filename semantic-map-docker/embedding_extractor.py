"""
This script is to save
embeddings from the last conv layer
of a ResNet, after L2-norm,
inspired by paper by Qi et al. (CVPR 2018)

The permutation and indexing part (not completed) tries to
reproduce the efforts by Amato et al. (SIGIR 2018)

"""

import torch
from data_loaders import img_preproc

#target_dim = 256 #2048 #512  # i.e. output size of the last layer kept in ResNet


def extract_embeddings(model, path_to_state, path_to_data, device, outp, transforms=None):

    model.load_state_dict(torch.load(path_to_state))

    model.eval()

    data = torch.load(path_to_data)

    embeddings = {}

    for img_id, img in data.items():

        #Applying same normalization as on a training forward pass
        img[0,:] = transforms(img[0,:].float())

        img = img.float().to(device)

        """
        base_embed = model.get_embedding(img) #shape: 1x2048

        # permute it
        embedding = permute(base_embed)

        if K is not None:

            embedding = truncate(embedding, K)

        # associate with codewords
        embedding = toText(embedding)
        """
        #Save embedding in dictionary under unique id
        embeddings[img_id] = model.get_embedding(img) #embedding


    #Save dictionary locally, as JSON file
    with open(outp, mode='wb') as outf:
        torch.save(obj=embeddings, f=outf)

    print("Trained embeddings saved under %s" % outp)


def query_embedding(model, path_to_state, path_to_img,  device, transforms=None):

    model.load_state_dict(torch.load(path_to_state))
    model.eval()

    #read image
    img = torch.from_numpy(img_preproc(path_to_img))

    img[0, :] = transforms(img[0, :].float())

    img = img.float().to(device)

    #Extract embedding for the query image
    return model.get_embedding(img)


def permute(embedding):

    #Returns the input embedding permutation
    return torch.argsort(embedding, dim=1, descending=True)

def truncate(embedding, K=100):

    return

def toText(embedding):

    # Assigns codewords to the input embedding

    return