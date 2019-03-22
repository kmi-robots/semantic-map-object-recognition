import torch
from torch import nn
from collections import OrderedDict
import torchvision.models as models


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


from rpooling import GeM, L2N

#Contrastive loss as defined in https://github.com/adambielski/siamese-triplet/
class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()

class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)

        return losses.mean() if size_average else losses.sum()



class NetForEmbedding(nn.Module):

    """
    Pre-trained Net used to generate img embeddings
    on each siamese pipeline
    """

    def __init__(self, feature_extraction=False, custom_pool=False):

        super().__init__()
        self.resnet = models.resnet50(pretrained=True)
        # Drop pooling + last FC layer

        if custom_pool:

           self.mod_resnet = nn.Sequential(*list(self.resnet.children())[:-2])  #-2 for GeM
           self.pooling = GeM() # no whitening for now, all params to default

        else:
            self.mod_resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        self.norm = L2N()

        if feature_extraction:

            for param in self.mod_resnet.parameters():

                param.requires_grad = False

    def forward(self, x, custom_pool=False):

        if custom_pool:

            x = self.norm(self.pooling(self.mod_resnet(x))).squeeze(-1).squeeze(-1)

            return x #.permute(1,0)

        else:
            x = self.mod_resnet(x)
            #print(self.mod_resnet(x).shape)
            return x.view(x.size(0), -1)  #self.norm(self.mod_resnet(x)).squeeze(-1).squeeze(-1)


#L2 norm and weight norm on pre-training as derived by paper
# by Qi et al. (CVPR 2018) - Google, but on ResNet here

class ResSiamese(nn.Module):


    """
    Overall structure of a siamese again,
    but each pipeline here is a pre-trained ResNet
    to exploit transfer learning
    - if a False flag is passed, the network will be fine-tuned instead
    """

    def __init__(self, feature_extraction=False, p=0.5, norm=True, scale=True):

        super().__init__()

        self.embed = NetForEmbedding(feature_extraction)
        #self.linear1 = nn.Linear(512, 512)


        self.fc = nn.Sequential(OrderedDict({
            'linear_1': nn.Linear(2048, 512),
            'relu_1': nn.ReLU(),
            'linear_2': nn.Linear(512, 512),
            'relu_2': nn.ReLU()
        }))

        self.drop = nn.Dropout(p=p)

        self.linear3 = nn.Linear(256, 2, bias=False)

        self.linear1 = nn.Linear(2048, 256) #set as weight imprinting example
        self.linear2 = nn.Linear(2048,2)  #(512, 2)

        self.norm = norm
        self.scale = scale
        self.s = nn.Parameter(torch.FloatTensor([10]))

    def forward_once(self, x):

        x = self.embed(x)
        #Flatten + FC + dropout
        if self.norm:

            x = self.l2_norm(x)

        if self.scale:
            x = self.s * x

        return self.linear1(x) #x.view(x.size(0), -1) #self.fc(x.view(x.size(0), -1)) #self.drop(self.linear2(x))

    def forward(self, data):

        #res1 = self.forward_once(data[0])
        #res2 = self.forward_once(data[1])
        res = torch.abs(self.forward_once(data[1]) - self.forward_once(data[0]))
        #self.drop(self.linear2(res))

        return self.linear3(res)

    def l2_norm(self, x):
        input_size = x.size()
        buffer = torch.pow(x, 2)

        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)

        _output = torch.div(x, norm.view(-1, 1).expand_as(x))

        output = _output.view(input_size)

        return output


    def weight_norm(self):

        w = self.linear2.weight.data
        norm = w.norm(p=2, dim=1, keepdim=True)

        self.linear2.weight.data = w.div(norm.expand_as(w))



    def get_embedding(self, x):

        x = self.embed(x)

        return self.l2_norm(x)


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