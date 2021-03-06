import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
import torchvision.models as models




from rpooling import GeM, L2N


#Contrastive loss as defined in https://github.com/adambielski/siamese-triplet/
class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin=2.0):
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

    def __init__(self, margin=2.0):
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

        if custom_pool:
           # Drop pooling + last FC layer

           self.mod_resnet = nn.Sequential(*list(self.resnet.children())[:-2])  #-2 for GeM
           self.pooling = GeM() # no whitening for now, all params to default

        else:
            #Only drop last FC, keep pre-trained avgpool
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


class ResSiamese(nn.Module):


    """
    Overall structure of a siamese again,
    but each pipeline here is a pre-trained ResNet
    to exploit transfer learning
    - if a False flag is passed, the network will be fine-tuned instead
    """

    def __init__(self, feature_extraction=False, p=0.5, norm=True, scale=True, stn=False):

        super().__init__()

        self.embed = NetForEmbedding(feature_extraction)
        #self.linear1 = nn.Linear(512, 512)


        self.fc = nn.Sequential(OrderedDict({
            'linear_1': nn.Linear(2048, 512),
            'relu_1': nn.ReLU(),
            'linear_2': nn.Linear(512, 512),
            'relu_2': nn.ReLU()
        }))



        self.linear3 = nn.Linear(256, 2) #, bias=False) # No bias used in the classifier layer of weight imprinting

        self.linear1 = nn.Linear(2048, 256) #set as weight imprinting example
        self.linear2 = nn.Linear(2048,2)  #(512, 2)

        self.norm = norm
        self.relu= nn.ReLU()
        self.scale = scale
        self.s = nn.Parameter(torch.FloatTensor([10]))
        self.stn_flag = stn

        if self.stn_flag:

            p = 0.0

        self.drop = nn.Dropout(p=p)
        self.drop2d = nn.Dropout2d(p=0.2)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.BatchNorm2d(10),
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 52 * 52, 32),
            nn.ReLU(True),
            nn.Dropout(p=0.2),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[3].weight.data.zero_()
        self.fc_loc[3].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 52 * 52)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward_once(self, x):

        if self.stn_flag:

            #If Spatial Transformer Net is activated
            x = self.stn(x)


        x = self.embed(x)

        """
        if self.norm:

            x = self.l2_norm(x)

        if self.scale:

            x = self.s * x

        """
        return self.l2_norm(x) # #x.view(x.size(0), -1) #self.fc(x.view(x.size(0), -1)) #self.drop(self.linear2(x))


    def forward(self, data):

        x0 = self.forward_once(data[0])
        x1 = self.forward_once(data[1])
        res = torch.abs(x1 - x0)

        res = self.drop(self.relu(self.linear1(res)))

        return self.linear3(self.drop(res)), x0, x1

    def l2_norm(self, x):

        input_size = x.size()

        buffer = torch.pow(x, 2)

        normp = torch.sum(buffer, 1).add_(1e-10)

        norm = torch.sqrt(normp)

        _output = torch.div(x, norm.view(-1, 1).expand_as(x))

        output = _output.view(input_size)

        return output

    """
    def weight_norm(self):

        w = self.linear2.weight.data

        norm = w.norm(p=2, dim=1, keepdim=True)

        self.linear2.weight.data = w.div(norm.expand_as(w))
    """

    def get_embedding(self, x):

        x = self.embed(x)

        return self.l2_norm(x)


class NNet(nn.Module):


    """
    Trying to replicate NNet by Zeng et al. (2018)
    - original paper model was in Torch Lua
    """

    def __init__(self, feature_extraction=False, p=0.5, norm=True):

        super().__init__()

        self.embed = NetForEmbedding(feature_extraction)
        self.embed2 = NetForEmbedding(feature_extraction=True) #Extracting feature on the synthetic-data branches
        self.embed3 = NetForEmbedding(feature_extraction=True)
        #self.l2dis = nn.PairwiseDistance()
        self.norm = norm

        self.drop = nn.Dropout(p=p)
        self.drop2d = nn.Dropout2d(p=0.2)


    def forward_once(self, x):

        x = self.embed(x)

        return F.normalize(x) # #x.view(x.size(0), -1) #self.fc(x.view(x.size(0), -1)) #self.drop(self.linear2(x))

    def forward_branch2(self, x):

        x = self.embed2(x)

        return F.normalize(x)

    def forward_branch3(self, x):

        x = self.embed3(x)

        return F.normalize(x)


    def forward(self, data):

        x0 = self.forward_once(data[0])
        x1 = self.forward_branch2(data[1])
        x2 = self.forward_branch2(data[2])

        return x0, x1, x2


    def get_embedding(self, x):

        x = self.embed(x)

        return F.normalize(x)


class KNet(nn.Module):
    """
    Trying to replicate KNet by Zeng et al. (2018)
    - original paper model was in Torch Lua
    https://github.com/andyzeng/arc-robot-vision/tree/master/image-matching
    """

    def __init__(self, feature_extraction=False, p=0.5, norm=True, num_classes=10):
        super().__init__()

        self.embed = NetForEmbedding(feature_extraction)
        self.embed2 = NetForEmbedding(feature_extraction=True) #Extracting feature on the synthetic-data branches
        self.embed3 = NetForEmbedding(feature_extraction=True)
        self.l2dis = nn.PairwiseDistance()
        self.l2dis_neg = nn.PairwiseDistance()

        self.norm = norm

        self.drop = nn.Dropout(p=p)
        self.drop2d = nn.Dropout2d(p=0.2)

        self.classifier = nn.Sequential(

            nn.Linear(2048,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512,128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes)
        )


    def forward_once(self, x):
        x = self.embed(x)

        return F.normalize(x)  # #x.view(x.size(0), -1) #self.fc(x.view(x.size(0), -1)) #self.drop(self.linear2(x))

    def forward_branch2(self, x):
        x = self.embed2(x)

        return F.normalize(x)

    def forward_branch3(self, x):
        x = self.embed3(x)

        return F.normalize(x)

    def forward(self, data):
        x0 = self.forward_once(data[0])
        x1 = self.forward_branch2(data[1])
        x2 = self.forward_branch2(data[2])

        #dis_pos = self.l2dis(x0, x1)
        #dis_neg = self.l2dis_neg(x0, x2)

        #joint_node = torch.cat(dis_pos, dis_neg)

        return x0, x1, x2, self.classifier(x0)


    def get_embedding(self, x):
        x = self.embed(x)

        return F.normalize(x)



class ImprintedKNet(nn.Module):

    """
    Adding weight imprinting on our implementation of KNet by Zeng et al. (2018)
    - original paper model was in Torch Lua
    https://github.com/andyzeng/arc-robot-vision/tree/master/image-matching
    """

    def __init__(self, feature_extraction=False, p=0.5, norm=True, num_classes=10):
        super().__init__()

        self.embed = NetForEmbedding(feature_extraction)
        self.embed2 = NetForEmbedding(feature_extraction)
        self.embed3 = NetForEmbedding(feature_extraction)

        self.norm = norm

        self.fcs1 = nn.Sequential(

            nn.Linear(2048,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512,128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.5)

        )

        self.fc2 = nn.Linear(128, num_classes, bias=False) #Removed bias from last layer

        self.scale = nn.Parameter(torch.FloatTensor([10]))


    def forward_once(self, x):
        x = self.embed(x)

        return F.normalize(x)  # #x.view(x.size(0), -1) #self.fc(x.view(x.size(0), -1)) #self.drop(self.linear2(x))

    def forward_branch2(self, x):
        x = self.embed2(x)

        return F.normalize(x)

    def forward_branch3(self, x):
        x = self.embed3(x)

        return F.normalize(x)

    def forward(self, data):

        x0 = self.forward_once(data[0])
        x1 = self.forward_branch2(data[1])
        x2 = self.forward_branch2(data[2])

        res = self.scale * self.l2_norm(self.fcs1(x0))

        return x0, x1, x2, self.fc2(res)


    def get_embedding(self, x):
        x = self.embed(x)

        return F.normalize(x)

    def extract(self, x):

        x = self.get_embedding(x)
        return self.scale *self.l2_norm(self.fcs1(x))

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



class ResTwoBranch(nn.Module):


    """
    Same as ResSiamese, except weights are not
    shared in the 2 ConvNet branches
    Without L2 norm on branch output embeddings
    """

    def __init__(self, feature_extraction=False, p=0.5, norm=True, scale=True, stn=False):

        super().__init__()

        self.embed = NetForEmbedding(feature_extraction)
        #self.linear1 = nn.Linear(512, 512)
        self.embed_branch2 = NetForEmbedding(feature_extraction)

        self.fc = nn.Sequential(OrderedDict({
            'linear_1': nn.Linear(2048, 512),
            'relu_1': nn.ReLU(),
            'linear_2': nn.Linear(512, 512),
            'relu_2': nn.ReLU()
        }))


        self.linear3 = nn.Linear(256, 2) #, bias=False) # No bias used in the classifier layer of weight imprinting

        self.linear1 = nn.Linear(2048, 256) #set as weight imprinting example
        self.linear2 = nn.Linear(2048,2)  #(512, 2)

        self.norm = norm
        self.relu= nn.ReLU()
        self.scale = scale
        self.s = nn.Parameter(torch.FloatTensor([10]))
        self.stn_flag = stn

        if self.stn_flag:

            p = 0.0

        self.drop = nn.Dropout(p=p)
        self.drop2d = nn.Dropout2d(p=0.2)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.BatchNorm2d(10),
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 52 * 52, 32),
            nn.ReLU(True),
            nn.Dropout(p=0.2),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[3].weight.data.zero_()
        self.fc_loc[3].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def stn(self, x):

        xs = self.localization(x)
        xs = xs.view(-1, 10 * 52 * 52)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward_once(self, x):

        if self.stn_flag:

            #If Spatial Transformer Net is activated
            x = self.stn(x)


        x = self.embed(x)

        """
        if self.norm:

            x = self.l2_norm(x)

        if self.scale:

            x = self.s * x

        """
        return x # #x.view(x.size(0), -1) #self.fc(x.view(x.size(0), -1)) #self.drop(self.linear2(x))

    def forward_branch2(self, x):

        if self.stn_flag:

            #If Spatial Transformer Net is activated
            x = self.stn(x)


        x = self.embed_branch2(x)

        return x # #x.view(x.size(0), -1) #self.fc(x.view(x.size(0), -1)) #self.drop(self.linear2(x))


    def forward(self, data):

        x0 = self.forward_once(data[0])
        x1 = self.forward_branch2(data[1])
        res = torch.abs(x1 - x0)

        res = self.drop(self.relu(self.linear1(res)))

        return self.linear3(self.drop(res)), x0, x1


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

        #self.normxcorrconv = normxcorrConv()

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