from __future__ import division
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import pdb

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models
import os

from imageio import imread # Image Reading
import matplotlib.pyplot as plt

import numpy as np
import itertools
import scipy.sparse
import cProfile
import re
from numpy.lib.stride_tricks import as_strided
unloader = transforms.ToPILImage()


import logging
import cv2
import numpy as np
from numpy.lib.stride_tricks import as_strided
import scipy.sparse
import scipy.sparse.linalg
import torch
import torch.nn as nn
from imageio import imread 
class GramMatrix(nn.Module):
    
    def forward(self, input):
        a, b, c, d = input.size()
        features = input.view(a * b, c * d)
        G = torch.mm(features, features.t())
        
        return G.div(a * b * c * d)

import torch.optim as optim
import torchvision.models as models
def _rolling_block(A, block=(3, 3)):
    """Applies sliding window to given matrix."""
    shape = (A.shape[0] - block[0] + 1, A.shape[1] - block[1] + 1) + block
    strides = (A.strides[0], A.strides[1]) + A.strides
    return as_strided(A, shape=shape, strides=strides)
def compute_laplacian(img, mask=None, eps=10**(-7), win_rad=1):
    """Computes Matting Laplacian for a given image.
    Args:
        img: 3-dim numpy matrix with input image
        mask: mask of pixels for which Laplacian will be computed.
            If not set Laplacian will be computed for all pixels.
        eps: regularization parameter controlling alpha smoothness
            from Eq. 12 of the original paper. Defaults to 1e-7.
        win_rad: radius of window used to build Matting Laplacian (i.e.
            radius of omega_k in Eq. 12).
    Returns: sparse matrix holding Matting Laplacian.
    """

    win_size = (win_rad * 2 + 1) ** 2
    h, w, d = img.shape
    # Number of window centre indices in h, w axes
    c_h, c_w = h - 2 * win_rad, w - 2 * win_rad
    win_diam = win_rad * 2 + 1

    indsM = np.arange(h * w).reshape((h, w))
    ravelImg = img.reshape(h * w, d)
    win_inds = _rolling_block(indsM, block=(win_diam, win_diam))

    win_inds = win_inds.reshape(c_h, c_w, win_size)
    if mask is not None:
        mask = cv2.dilate(
            mask.astype(np.uint8),
            np.ones((win_diam, win_diam), np.uint8)
        ).astype(np.bool)
        win_mask = np.sum(mask.ravel()[win_inds], axis=2)
        win_inds = win_inds[win_mask > 0, :]
    else:
        win_inds = win_inds.reshape(-1, win_size)

    
    winI = ravelImg[win_inds]

    win_mu = np.mean(winI, axis=1, keepdims=True)
    win_var = np.einsum('...ji,...jk ->...ik', winI, winI) / win_size - np.einsum('...ji,...jk ->...ik', win_mu, win_mu)

    inv = np.linalg.inv(win_var + (eps/win_size)*np.eye(3))

    X = np.einsum('...ij,...jk->...ik', winI - win_mu, inv)
    vals = np.eye(win_size) - (1.0/win_size)*(1 + np.einsum('...ij,...kj->...ik', X, winI - win_mu))

    nz_indsCol = np.tile(win_inds, win_size).ravel()
    nz_indsRow = np.repeat(win_inds, win_size).ravel()
    nz_indsVal = vals.ravel()
    L = scipy.sparse.coo_matrix((nz_indsVal, (nz_indsRow, nz_indsCol)), shape=(h*w, h*w))
    
    rowIdx = nz_indsCol[nz_indsVal!=0]; # Discard index associated to zero value
    colIdx = nz_indsRow[nz_indsVal!=0];   # Discard index associated to zero value
    nz_indsVal = nz_indsVal[nz_indsVal!=0];     # Discard zero value 
    i = torch.LongTensor(np.vstack([rowIdx,colIdx]))
    v = torch.FloatTensor(nz_indsVal)
    M = torch.sparse.FloatTensor(i,v,torch.Size([h*w,h*w]))
    return M
import numpy as np
class StyleCNN(object):
    def __init__(self, style, content, pastiche):
        super(StyleCNN, self).__init__()
        
        self.style = style
        self.content = content
        # print(content.shape)
        self.pastiche = nn.Parameter(pastiche.data)
        
        self.content_layers = ['conv_4']
        self.style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        self.content_weight = 1
        self.style_weight = 1000

        img_array = np.array(imread("images/in1.jpg")).astype("float")
        self.matting = compute_laplacian(img_array).cuda() 

        self.loss_network = models.vgg19(pretrained=True)
        self.gram = GramMatrix()
        self.loss = nn.MSELoss()
        self.optimizer = optim.LBFGS([self.pastiche])
        
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.loss_network.cuda()
            self.gram.cuda()

    def train(self):
        def closure():
            self.optimizer.zero_grad()
          
            pastiche = self.pastiche.clone()
            pastiche.data.clamp_(0, 1)
            content = self.content.clone()
            style = self.style.clone()
            
            content_loss = 0
            style_loss = 0
            # print(pastiche)
            # print(self.matting)
            photo_loss = PhotorealismLoss.apply(pastiche, self.matting)
            i = 1
            not_inplace = lambda layer: nn.ReLU(inplace=False) if isinstance(layer, nn.ReLU) else layer
            for layer in list(self.loss_network.features):
                layer = not_inplace(layer)
                if self.use_cuda:
                    layer.cuda()
                    
                pastiche, content, style = layer.forward(pastiche), layer.forward(content), layer.forward(style)
                
                if isinstance(layer, nn.Conv2d):
                    name = "conv_" + str(i)
                    
                    if name in self.content_layers:
                        content_loss += self.loss(pastiche * self.content_weight, content.detach() * self.content_weight)
                    
                    if name in self.style_layers:
                        pastiche_g, style_g = self.gram.forward(pastiche), self.gram.forward(style)
                        style_loss += self.loss(pastiche_g * self.style_weight, style_g.detach() * self.style_weight)
                
                if isinstance(layer, nn.ReLU):
                    i += 1
            
            total_loss = content_loss + style_loss + photo_loss
            total_loss.backward()
            
            return total_loss
        
        self.optimizer.step(closure)
        return self.pastiche

class PhotorealismLoss(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, ml):
                ctx.save_for_backward(input)
                ctx.ml = ml.clone()
                # print(input.shape)
                input = input.clone()
                input_r = input[0,0,:,:].view(1,-1)
                input_g = input[0,1,:,:].view(1,-1)
                input_b = input[0,2,:,:].view(1,-1)
                # print(input.shape)
                r = torch.matmul(input_r, ml.matmul(input_r.t()))
                g = torch.matmul(input_g, ml.matmul(input_g.t()))
                b = torch.matmul(input_b, ml.matmul(input_b.t()))
                a = torch.cat([r,g,b]).view(3).mean()
                # print(a)
                return torch.FloatTensor([a]).cuda() 
        @staticmethod
        def backward(ctx, grad_output):
                input = ctx.saved_variables
                # print(ctx.ml)
                # print(input)
                input = input[0].data
                # print(input[0].shape)
                img_dim_1 = input.shape[2]
                img_dim_2 = input.shape[3]
                ml = ctx.ml.clone()
                input_r = input[0,0,:,:].view(1,-1)
                # print(input_r)
                input_g = input[0,1,:,:].view(1,-1)
                input_b = input[0,2,:,:].view(1,-1)
                grad = torch.cat([2 * ml.matmul(input_r.t()).view(1,img_dim_1,img_dim_2),
                    2 * ml.matmul(input_g.t()).view(1,img_dim_1,img_dim_2),
                    2 * ml.matmul(input_b.t()).view(1,img_dim_1,img_dim_2)]).view(1, 3, img_dim_1, img_dim_2)*10 /(img_dim_1*img_dim_2)
                # print(grad)
                # print(a.shape)
                return Variable(grad), None

import torchvision.transforms as transforms
from torch.autograd import Variable

from PIL import Image
import scipy.misc
import matplotlib.pyplot as plt
imsize = 256
def Matting(img,winSize=1,eps=1e-9):
    ''' Compute the Matting Laplacina Matrix of image img
            input, img: input Image
            winSize, window size for constructing matting laplacian, default =1 
            eps: regularization parameter, default = 1e-9
        output,
            M: the matting laplacian matrix (sparse)
            to output the dense matrix. use M.toarray()
    '''
    s0=img.shape[0] 
    s1=img.shape[1]
    nPixels =s0*s1               # Number of pixels
    winNumel = (winSize*2+1)**2  # Number of elements in a 
    winDiam = winSize*2+1
    I = np.reshape(img,(-1,3))   # Flatten Image
    # IT = torch.from_numpy(I)   # pytorch     
                                 
    # Compute Indexs in each window, each row stores the indexs of elements in the same window
    # Only "full" windows are stored 
    shape = (s0 - winDiam + 1, s1 - winDiam + 1) + (winDiam,winDiam)
    idx = np.arange(s0 * s1).reshape((s0, s1))
    strides = (idx.strides[0], idx.strides[1]) + idx.strides
    winIdxs = as_strided(idx, shape=shape, strides=strides)
    winIdxs = winIdxs.reshape((winIdxs.shape[0]*winIdxs.shape[1],winIdxs.shape[2]*winIdxs.shape[3]))
    
    #winIdxs = np.stack([neigh for neigh in wins(winSize,s0,s1) if len(neigh)==(2*winSize+1)**2])
    Ik = I[winIdxs]              # Pixels value in each windows. 
                                 # I[i,k,:] returns the RGB of kth element in ith window
    # winIdxsT=torch.LongTensor(torch.from_numpy(winIdxs))
    # IkT = torch.FloatTensor(winIdxs.shape[0],winIdxs.shape[1],3).zero_()
    # for k in range(winIdxsT.shape[1]):
    #    IkT[k] = IT[winIdxsT[0]]
    
    muk = np.mean(Ik,axis=1,keepdims=True) # Mean RGB pixel intensity of each window 
                                 # muk[i,0,:] returns the average RGB intensity of ith window
    # mukT = IkT.mean(1,keepdim=True)
        
    varMtxs = np.zeros((Ik.shape[0],3,3)); # 3x3 Covariance matrices in each window 
    varMtxs = np.einsum('Cni,Cnj->Cij',Ik-muk,Ik-muk)/Ik.shape[1] # varMtxs[i,:,:] returns the 3x3 CovMtx in ith window
    # varMtxsT = torch.bmm((IkT-mukT).transpose(1,2),(IkT-mukT))
    
    invs = np.linalg.inv(varMtxs + (eps/winNumel)*np.eye(3))      # Regularized inverses of varMtxs, 
    
    quads = np.zeros((Ik.shape[0],winNumel,winNumel)); # value of Quadratic terms in the closed form matting laplacian
    quads = np.einsum('Cia,Cab,Cjb->Cij',Ik-muk,invs,Ik-muk)

    rowIdx = np.repeat(winIdxs,winNumel,axis=1).ravel() # slow ticking indices in each window 
    colIdx = np.tile(winIdxs,winNumel).ravel()          # fast ticking indices in each window 
    vals = ((rowIdx==colIdx).astype('float'))-(1.0/(winNumel)*(1+quads.ravel())) 
                                              # Incremental Values associate to each (rowIdx,colIdx) element in M
#   M=scipy.sparse.coo_matrix((vals, (rowIdx, colIdx)), shape=(nPixels, nPixels)) output Scipy sprase
    
    rowIdx = rowIdx[vals!=0]; # Discard index associated to zero value
    colIdx = vals[vals!=0];   # Discard index associated to zero value
    vals = vals[vals!=0];     # Discard zero value 
    i = torch.LongTensor(np.vstack([rowIdx,colIdx]))
    v = torch.FloatTensor(vals)
    M = torch.sparse.FloatTensor(i,v,torch.Size([nPixels,nPixels]))
    return M
loader = transforms.Compose([
             transforms.ToTensor()
         ])

unloader = transforms.ToPILImage()

def image_loader(image_name):
    image = Image.open(image_name)
    image = Variable(loader(image))
    image = image.unsqueeze(0)
    return image
  
def save_image(input, path):
    image = input.data.clone().cpu()
    image = image.view(3, 444, 444)
    image = unloader(image)
    plt.imsave(path, image)

import torch.utils.data
import torchvision.datasets as datasets



def main():
    # CUDA Configurations
    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    # Content and style
    style = image_loader("images/style1.jpg").type(dtype)
    content = image_loader("images/in1.jpg").type(dtype)

    pastiche = image_loader("images/in1.jpg").type(dtype)
    pastiche.data = torch.randn(pastiche.data.size()).type(dtype)
    print(pastiche.shape)
    num_epochs = 41
    style_cnn = StyleCNN(style, content, pastiche)
    
    for i in range(num_epochs):
        pastiche = style_cnn.train()
    
        if i % 10 == 0:
            print("Iteration: %d" % (i))
            
            path = "outputs/new%d.png" % (i)
            pastiche.data.clamp_(0, 1)
            save_image(pastiche, path)

main()