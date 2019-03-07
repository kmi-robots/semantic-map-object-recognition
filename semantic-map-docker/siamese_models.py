import torch
from torch import nn
from collections import OrderedDict
import time
import torchvision.models as models
import torch.nn.functional as F


class SimplerNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.sequential = nn.Sequential(OrderedDict({
            'conv_1': nn.Conv2d(3, 20, kernel_size=5),
            'relu_1': nn.ReLU(),
            'maxpool_1': nn.MaxPool2d(2),
            'conv_2': nn.Conv2d(20, 25, kernel_size=5),
            'relu_2': nn.ReLU(),
            'maxpool_2': nn.MaxPool2d(2),
            'conv_3': nn.Conv2d(25, 30, kernel_size=5),
            'relu_3': nn.ReLU(),
            'maxpool_3': nn.MaxPool2d(2)
        }))

        self.linear1 = nn.Linear(16 * 4 * 30, 500)

        self.linear2 = nn.Linear(500, 2)

    def forward_once(self, x):
        x = self.sequential(x)

        x = x.view(x.size(0), -1)

        return self.linear1(x)

    def forward(self, data):
        res = torch.abs(self.forward_once(data[1]) - self.forward_once(data[0]))
        res = self.linear2(res)
        return res





class ResSiamese(nn.Module):


    """
    Overall structure of a siamese again,
    but each pipeline here is a pre-trained ResNet
    to exploit transfer learning
    - if a False flag is passed, the network will be fine-tuned instead
    """

    def __init__(self, feature_extraction=False):

        super().__init__()
        self.resnet = models.resnet18(pretrained=True)
        #Drop last FC layer
        self.mod_resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        if feature_extraction:

            for param in self.mod_resnet.parameters():

                param.requires_grad = False

        self.linear1 = nn.Linear(512, 512)

        self.linear2 = nn.Linear(512, 2)

    def forward_once(self, x):

        x = self.mod_resnet(x)

        x = x.view(x.size(0), -1)

        return self.linear1(x)

    def forward(self, data):
        res = torch.abs(self.forward_once(data[1]) - self.forward_once(data[0]))
        res = self.linear2(res)
        return res


#Reproducing NormXCorr model by Submariam et al. (NIPS 2016)
from torch.autograd import Function

import ncc  #Our built extension

class normxcorr(Function):

    @staticmethod
    def forward(ctx, X_, Y_, patch_size=5, stride=1, epsilon=0.01):

        ctx.saved_for_backward = [X_, Y_]
        ctx.patch_size = patch_size
        ctx.stride = stride
        ctx.epsilon = epsilon

        return ncc.forward(X_, Y_, ctx.patch_size, ctx.stride, ctx.epsilon)

    @staticmethod
    def backward(ctx, grad_output):

        input1, input2 = ctx.saved_for_backward

        #grad_input1 = torch.zeros(input1.size()).to(device)
        #grad_input2 = torch.zeros(input2.size()).to(device)

        nan_idxs = torch.nonzero(torch.isnan(grad_output))

        # Print warning in case any NaN is found
        if nan_idxs.numel() != 0:
            print("NaN value found in grad_output ")

        # grad_input = [derivate forward(input) wrt parameters] * grad_output
        grad_input1, grad_input2 = ncc.backward(input1,
                                                 input2,
                                                grad_output,
                                                ctx.patch_size,
                                                ctx.stride,
                                                ctx.epsilon)



        nan_idxs = torch.nonzero(torch.isnan(grad_input1))

        # Print warning in case any is found
        if nan_idxs.numel() != 0:
            print("NaN value found in grad_input1 ")

        nan_idxs = torch.nonzero(torch.isnan(grad_input2))

        # Print warning in case any is found
        if nan_idxs.numel() != 0:
            print("NaN value found in grad_input2")

        return grad_input1, grad_input2


#Trying version that exploits Conv2D
class normxcorrConv(nn.Module):

    def __init__(self):
        super(normxcorrConv, self).__init__()

    def forward_once(self, X, Y):


        return X

    def pad(self, X, Y, d=2):

        return F.pad(X, (d, d, d, d)), F.pad(Y, (d, d, 2 * d, 2 * d))

    def forward(self, input):

        X = input[0]
        Y = input[1]

        X_pad, Y_pad = self.pad(X,Y)

        all_Es = X_pad.unfold(2, 5, 1).unfold(3, 5, 1)

        print(X.shape)

        return self.forward_once(X,Y)


# The whole feed-forward architecture
class NCCNet(nn.Module):

    def __init__(self):


        super(NCCNet, self).__init__()
        """
        All variables defined here are ultimately
        passed as params to the optimizer

        """

        # Redefining first part as a sequential model
        self.sequential = nn.Sequential(OrderedDict({
            'conv_1': nn.Conv2d(3, 20, kernel_size=5),
            'relu_1': nn.ReLU(),
            'maxpool_1': nn.MaxPool2d(2),
            'conv_2': nn.Conv2d(20, 25, kernel_size=5),
            'relu_2': nn.ReLU(),
            'maxpool_2': nn.MaxPool2d(2)
        }))

        # self.normxcorr = NormXCorr().to(device)

        self.normrelu = nn.ReLU()

        self.dimredux = self.dimredux = nn.Sequential(

            nn.Conv2d(1500, 25, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(25, 25, kernel_size=3),
            nn.MaxPool2d(2)
        )

        self.normxcorrconv = normxcorrConv()

        self.linear1 = nn.Linear(25 * 17 * 5, 500)

        self.linear2 = nn.Linear(500, 2)

        self.softmax = nn.Softmax()

    def forward_once(self, x):
        return self.sequential(x)

    def forward(self, input):
        # burden_start = time.time()

        # Defines the two pipelines, one for each input, i.e., siamese-like
        # print("Passed first pipeline in %f seconds" % (time.time() - burden_start))

        #reset = time.time()

        res = normxcorr.apply(self.forward_once(input[0]), self.forward_once(input[1]))  # batch_size x 1500 x 37 x 12

        #print("Passed NormXCorr in %f seconds" % (time.time() - reset))
        #reset = time.time()

        #res2 = self.normxcorrconv(input)
        #print("Passed NormXCorr in %f seconds" % (time.time() - reset))

        #print(res[0,0,:])
        #print(res2[0,0,:])

        #print("Passed NormXCorr in %f seconds" % (time.time() - reset))

        # reset = time.time()

        res = self.normrelu(res)

        res = self.dimredux(res)  # batch_size x 25 x 17 x 5

        res = res.view(res.size()[0], -1)  # (batch_size. 2125) , i.e. flattened

        res = self.linear1(res)  # batch_size x 2121 x 500

        res = self.linear2(res)  # batch_size x 500 x 2

        # print("Passed through remaining layers in %f seconds" % (time.time() - reset))
        # print(res.shape)
        #res2 = self.softmax(res)  # Calculated in train loop later

        return res


def check_for_NaNs(model):

    model_dict = model.state_dict()

    for key in model_dict.keys():

        curr_tensor = model_dict[key]

        #print(curr_tensor)

        #Find indices of all NaNs, if any
        nan_idxs = torch.nonzero(torch.isnan(curr_tensor))

        #Print warning in case any is found
        if nan_idxs.numel() != 0:

            print("NaN value found in Tensor %s" % key)
            print(curr_tensor)