"""
Script used to augment the robot reference dataset
(for training examples only)
"""

import torchvision
from torchvision import transforms
import PIL
from PIL import Image
import argparse
import os


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('inpath', help='path containing all image subfolders to be augmented')
    parser.add_argument('num_samples', help='number of extra examples to generate', type=int)

    args = parser.parse_args()

    trans = transforms.Compose([
        #transforms.Resize((224,224)),
        #transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomRotation(20, resample=PIL.Image.BILINEAR)
        ])
        #transforms.ToTensor()])

    data =torchvision.datasets.ImageFolder(args.inpath,\
        transform=trans)


    for i in range(len(data)):

        curr_class = data.classes[data[i][1]]

        for _ in range(args.num_samples):

            data[i][0].save(os.path.join(args.inpath,curr_class, 'DA'+'_'+str(i)+'_'+str(_)+'.png'))
