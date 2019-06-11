"""
This script is to save
embeddings from the last conv layer
of a ResNet, after L2-norm,
inspired by paper by Qi et al. (CVPR 2018)

"""

import torch
from data_loaders import img_preproc
from siamese_models import ResSiamese
import torchvision

def extract_embeddings(model, path_to_state, path_to_data, train_img_folder, device, outp, transforms=None):

    model.load_state_dict(torch.load(path_to_state))
    model.eval()
    #data = torch.load(path_to_data)
    data = torchvision.datasets.ImageFolder(train_img_folder, \
                                            transform=transforms)

    embeddings = {}

    for i in range(len(data)):
    #for img_id, img in data.items():

        #Applying same normalization as on a training forward pass
        #img[0,:] = #transforms(img[0,:])
        img_id = data.classes[data[i][1]]
        img =  data[i][0]
        img = img.view(1, img.shape[0], img.shape[1], img.shape[2]).to(device)
        #img = img.float().to(device)

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


    with open(outp, mode='wb') as outf:
        torch.save(obj=embeddings, f=outf)

    print("Trained embeddings saved under %s" % outp)


def path_embedding(model, path_to_state, path_to_img,  device, transforms=None):

    model.load_state_dict(torch.load(path_to_state))
    model.eval()

    #read image
    img = img_preproc(path_to_img, transforms) #torch.from_numpy(img_preproc(path_to_img))
    img = img.view(1, img.shape[0], img.shape[1], img.shape[2]).to(device)
    #img[0, :] = transforms(img[0, :].float())

    #img = img.float().to(device)

    #Extract embedding for the query image
    return model.get_embedding(img)


def array_embedding(model, path_to_state, img_array,  device, transforms=None):
    """
    Same as path_embedding but requiring image array instead of path
    to image file as input
    """
    #model = ResSiamese(feature_extraction=True).to(device)
    model.load_state_dict(torch.load(path_to_state, map_location={'cuda:0': 'cpu'}))
    model.eval()

    #read image
    img = img_preproc(img_array, transforms, ros=True) #torch.from_numpy(img_preproc(img_array, ros=True))

    #img[0, :] = transforms(img[0, :].float())
    img = img.view(1, img.shape[0], img.shape[1], img.shape[2]).to(device)
    #img = img.float().to(device)

    #Extract embedding for the query image
    return model.get_embedding(img)


def base_embedding(path_to_img,  device, transforms=None):

    model = ResSiamese(feature_extraction=True).to(device)
    #model.load_state_dict(torch.load(path_to_state))
    model.eval()

    #read image
    img = img_preproc(path_to_img, transforms) #torch.from_numpy(img_preproc(path_to_img))
    img = img.view(1, img.shape[0], img.shape[1], img.shape[2]).to(device)
    #img[0, :] = transforms(img[0, :].float())

    #img = img.float().to(device)

    #Extract embedding for the query image
    return model.get_embedding(img)


