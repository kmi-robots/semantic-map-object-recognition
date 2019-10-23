"""
Region Pooling based on the code implemented here:
https://github.com/filipradenovic/cnnimageretrieval-pytorch
by Radenović F., Tolias G. et al. (2018)

"""
import torch
from torch import nn
from torch.nn.parameter import Parameter
import cirtorch_functional as LF


class GeM(nn.Module):

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return LF.gem(x, p=self.p, eps=self.eps)

    def __repr__(self):

        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' \
           + 'eps=' + str(self.eps) + ')'


class L2N(nn.Module):

    def __init__(self, eps=1e-6):
        super(L2N, self).__init__()
        self.eps = eps

    def forward(self, x):
        return LF.l2n(x, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'eps=' + str(self.eps) + ')'

"""
class Rpool(nn.Module):

    def __init__(self, rpool, whiten=None, L=3, eps=1e-6):
        super(Rpool,self).__init__()
        self.rpool = rpool
        self.L = L
        self.whiten = whiten
        self.norm = L2N()
        self.eps = eps

    def forward(self, x, aggregate=True):
        # features -> roipool
        o = LF.roipool(x, self.rpool, self.L, self.eps) # size: #im, #reg, D, 1, 1

        # concatenate regions from all images in the batch
        s = o.size()
        o = o.view(s[0]*s[1], s[2], s[3], s[4]) # size: #im x #reg, D, 1, 1

        # rvecs -> norm
        o = self.norm(o)

        # rvecs -> whiten -> norm
        if self.whiten is not None:
            o = self.norm(self.whiten(o.squeeze(-1).squeeze(-1)))

        # reshape back to regions per image
        o = o.view(s[0], s[1], s[2], s[3], s[4]) # size: #im, #reg, D, 1, 1

        # aggregate regions into a single global vector per image
        if aggregate:
            # rvecs -> sumpool -> norm
            #Last layer is L2 norm
            o = self.norm(o.sum(1, keepdim=False)) # size: #im, D, 1, 1

        return o

    def __repr__(self):
        return super(Rpool, self).__repr__() + '(' + 'L=' + '{}'.format(self.L) + ')'


"""