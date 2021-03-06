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


def _rolling_block(A, block=(3, 3)):
    """Applies sliding window to given matrix."""
    shape = (A.shape[0] - block[0] + 1, A.shape[1] - block[1] + 1) + block
    strides = (A.strides[0], A.strides[1]) + A.strides
    return as_strided(A, shape=shape, strides=strides)


def show_imgs(content, output, style):
    fig,axes = plt.subplots(1,3,figsize=(12,5),dpi=150)
    imgs=[content, output, style]
    titles=['Content', 'Output', 'Style']
    for i in range(3):
        axes[i].imshow(imgs[i])
        axes[i].set_title(titles[i])
        axes[i].axis("off")
    #plt.imsave("out1.png", output)
    plt.show()

loader = transforms.Compose([
    transforms.ToTensor()])  # transform it into a torch tensor

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
    
    #varMtxs = np.zeros((Ik.shape[0],3,3)); # 3x3 Covariance matrices in each window 
    #varMtxs = np.einsum('Cni,Cnj->Cij',Ik-muk,Ik-muk)/(Ik.shape[1]) # varMtxs[i,:,:] returns the 3x3 CovMtx in ith window
    #varMtxsT = torch.bmm((IkT-mukT).transpose(1,2),(IkT-mukT))
    # varMtxs[i,:,:] returns the 3x3 CovMtx in ith window
    varMtxs = np.einsum('Cni,Cnj->Cij',Ik-muk,Ik-muk,dtype='double')/(winNumel) 
    # all zero sigma will be come eye(3)
    varMtxs[np.sum(np.sum(varMtxs,axis=1),axis=1)==0]=np.eye(3)  
    # Regularized inverses of varMtxs,
    invs = np.linalg.inv(varMtxs + (eps/winNumel)*np.eye(3))       
    # value of Quadratic terms in the closed form matting laplacian
    quads = np.einsum('Cia,Cab,Cjb->Cij',Ik-muk,invs,Ik-muk,dtype='double')

    
    rowIdx = np.repeat(winIdxs,winNumel,axis=1).ravel() # slow ticking indices in each window 
    colIdx = np.tile(winIdxs,winNumel).ravel()          # fast ticking indices in each window 
    vals = ((rowIdx==colIdx).astype('double'))-(1.0/(winNumel)*(1+quads.ravel())) 
                                              # Incremental Values associate to each (rowIdx,colIdx) element in M
    #M = scipy.sparse.coo_matrix((vals, (rowIdx, colIdx)), shape=(nPixels, nPixels))# output Scipy sprase
    #return (M+M.transpose())/2 ## for numerical stability
    
    rowIdx = rowIdx[vals!=0]; # Discard index associated to zero value
    colIdx = vals[vals!=0];   # Discard index associated to zero value
    vals = vals[vals!=0];     # Discard zero value 
    i = torch.LongTensor(np.vstack([rowIdx,colIdx]))
    v = torch.FloatTensor(vals)
    M = torch.sparse.FloatTensor(i,v,torch.Size([nPixels,nPixels]))
    return (M+M.t())/2


def img_to_variable(img):
    img=torch.FloatTensor(img)
    if cuda:
        img=img.cuda()
    img=img.permute(2,0,1)
    return Variable(img).unsqueeze(0)/255

def variable_to_im(var):
    return (var[0].permute(1,2,0)).data.cpu().numpy()

class GramMatrix(nn.Module):

    def forward(self, input):
        a, b, c, d = input.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)

        features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

        G = torch.mm(features, features.t())  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(a * b * c * d)

class ContentLoss(nn.Module):

    def __init__(self, target, weight):
        super(ContentLoss, self).__init__()
        if weight==0:
            self.forward=ret_self
            self.backward=zero
            return
        # we 'detach' the target content from the tree used
        self.target = target.detach() * weight
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.weight = weight
        self.criterion = nn.MSELoss()

    def forward(self, input):
        self.loss = self.criterion(input * self.weight, self.target)
        self.output = input
        return self.output

    def backward(self, retain_graph=True):
        # pdb.set_trace()
        self.loss.backward(retain_graph=retain_graph)
        print(self.loss)
        return self.loss

def zero():
    return 0

def ret_self(x):
    return x

class StyleLoss(nn.Module):

    def __init__(self, target, weight):
        super(StyleLoss, self).__init__()
        if weight==0:
            self.forward=ret_self
            self.backward=zero
            return
        self.target = target.detach() * weight
        self.weight = weight
        self.gram = GramMatrix()
        self.criterion = nn.MSELoss()

    def forward(self, input):
        self.output = input.clone()
        self.G = self.gram(input)
        self.G.mul_(self.weight)
        self.loss = self.criterion(self.G, self.target)
        return self.output

    def backward(self, retain_graph=True):
        # pdb.set_trace()
        self.loss.backward(retain_graph=retain_graph)
        return self.loss

class PhotorealismLoss(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, ml, weight):
                ctx.save_for_backward(input)
                ctx.ml = ml.clone()
                input = input.clone()
                input_r = input[0,0,:,:].view(1,-1)
                input_g = input[0,1,:,:].view(1,-1)
                input_b = input[0,2,:,:].view(1,-1)
                ctx.weight = weight
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
                	2 * ml.matmul(input_b.t()).view(1,img_dim_1,img_dim_2)]).view(1, 3, img_dim_1, img_dim_2) * ctx.weight /(img_dim_2*img_dim_1)
                # print(grad)
                # print(a.shape)
                return Variable(grad), None, None

def PhotorealismLossTests():
        dtype = torch.FloatTensor
        x = Variable(torch.FloatTensor([1,1,1]).type(dtype), requires_grad = True)
        m = Variable(torch.ones([3,3]).type(dtype), requires_grad = False)
        lossLOL = PhotorealismLoss.apply(x, m)
        print(lossLOL)
        print(x.grad.data)
        lossLOL.backward()
        print(x.grad.data)

def to_sparse(x):
    """ converts dense tensor x to sparse format """
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())

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

vgg=None
class Net(nn.Module):
    def __init__(self, content, style, content_weights, style_weights):
        #pdb.set_trace()
        super(Net, self).__init__()

        self.vgg=vgg
        self.style_losses=[]
        self.content_losses=[]
        # self.x=Variable(content.data.clone().cuda(), requires_grad=True)
        self.x=Variable(torch.rand(content.data.shape).cuda(), requires_grad=True)
        #self.x=nn.Parameter(content.data).cuda()
        self.content_weights = content_weights
        self.style_weights = style_weights

        gram=GramMatrix().cuda()
        layer_num=0
        for i in range(len(vgg)):
            content=vgg[i](content).clone()
            style=vgg[i](style).clone()
            ## added by allen
            #interX = vgg[0](self.x)
            #for j in range(1,i+1):
            #    interX =  vgg[j](interX)
            ###
            ##x=vgg[i](x)
            #x = interX
            if isinstance(vgg[i], nn.Conv2d):
                if self.style_weights[layer_num] == 0:
                    self.style_losses.append("dummy holder")
                else:
                    style_loss=StyleLoss(gram(style),self.style_weights[layer_num]).cuda()
                    self.style_losses.append(style_loss)
                if self.content_weights[layer_num] == 0:
                    self.content_losses.append("dummy holder")
                else:
                    content_loss=ContentLoss(content, self.content_weights[layer_num]).cuda()
                    self.content_losses.append(content_loss)
                layer_num+=1

    def forward(self):
        vgg=self.vgg
        self.x.data.clamp_(0,1)
        x=self.x
        layer_num=0
        for i in range(len(vgg)):
            x=vgg[i](x)
            if isinstance(vgg[i], nn.Conv2d):
                if not isinstance(self.style_losses[layer_num], str):
                    x = self.style_losses[layer_num](x)
                if not isinstance(self.content_losses[layer_num], str):
                    x = self.content_losses[layer_num](x)
                layer_num+=1

    def backward(self):
        # pdb.set_trace()
        raise NotImplementedError()
        loss_score=0
        for loss in self.style_losses+self.content_losses:
            loss_score+=loss.backward()
        print(loss_score)
        return loss_score
            #loss.backward()


def image_loader(image_name):
    image = Image.open(image_name)
    image = Variable(loader(image))
    # fake batch dimension required to fit network's input dimensions
    image = image.unsqueeze(0)
    return image

def run(content_img, style_img, content_weight, style_weight, lr, fname, matting_laplacian = False, weight = 1, num_steps = 300):

    content_img_var=img_to_variable(content_img)
    style_img_var=img_to_variable(style_img)
    # print(content_img_var[0,0,:,:].view(-1).shape)
    # conv1_1, conv_2_1, conv_3_1, conv_4_1,conv_5_1
    net=Net(content_img_var, style_img_var, content_weight, style_weight)
    optimizer = optim.LBFGS([net.x])
    if cuda:
        net.cuda()
    step_num=num_steps

    if matting_laplacian:
        img_array = np.array(content_img).astype("float")
        img_matting_laplacian = compute_laplacian(img_array).cuda() 
    for i in range(200):
        def closure():
            optimizer.zero_grad()
            net.forward()
            total_loss = 0
            for loss in net.content_losses + net.style_losses:
                if not isinstance(loss, str):
                    total_loss  = total_loss + loss.loss
            if matting_laplacian:
                matting_laplacian_loss = PhotorealismLoss.apply(net.x, img_matting_laplacian, weight)
                total_loss = total_loss + matting_laplacian_loss
            total_loss.backward()
            return total_loss
        optimizer.step(closure)
        net.x.data.clamp_(0,1)
        # loss=net.backward()
        # if matting_laplacian:
        #     matting_laplacian_loss = PhotorealismLoss.apply(net.x, img_matting_laplacian)
        #     matting_laplacian_loss.backward(torch.ones(content_img.shape))  
        # optimizer.step()
        # net.x.data.clamp_(0,1) 
        if i%10 == 0:
            print("current progress: " + str(i))
            plt.imsave('report_out/'+fname+str(style_weight[0]) + '_' + str(weight) + '_' + str(i)+'.jpg', variable_to_im(net.x))

    # net.x.data.clamp_(0,1) 
    #show_imgs(content_img, variable_to_im(net.x), style_img)
    # # plt.imsave('test_output/'+fname+'.jpg', variable_to_im(net.x))
    # for i in range(200):
    #     def closure():
    #         optimizer.zero_grad()
    #         # net.forward()
    #         matting_laplacian_loss = PhotorealismLoss.apply(net.x, img_matting_laplacian)
    #         total_loss = matting_laplacian_loss
    #         total_loss.backward()
    #         print(total_loss)
    #         return total_loss
    #     optimizer.step(closure)
    #     net.x.data.clamp_(0,1)
    #     # loss=net.backward()
    #     # if matting_laplacian:
    #     #     matting_laplacian_loss = PhotorealismLoss.apply(net.x, img_matting_laplacian)
    #     #     matting_laplacian_loss.backward(torch.ones(content_img.shape))  
    #     # optimizer.step()
    #     # net.x.data.clamp_(0,1) 
    #     if i%10 == 0:
    #         print("current progress: " + str(i))
    #         plt.imsave('out_last_resort/'+fname+"photo" + "_" + str(style_weight[0]) + '_' + str(weight) + '_' + str(i)+'.jpg', variable_to_im(net.x))
if __name__=='__main__':
    os.system('rm -rf output')
    os.mkdir('output')
    cuda = torch.cuda.is_available()
    style_weight = torch.ones(16).cuda()
    content_weight = torch.zeros(16).cuda()
    content_img = imread('./images/in0.jpg')
    style_img = imread('./images/style0.png')
    # PhotorealismLossTests()
    vgg=list(models.vgg19(pretrained=True).features.cuda())

    style_weight[13] = 0
    style_weight[14] = 0
    style_weight[15] = 0
    
    content_weight[7] = 1
    content_weight[8] = 1
    content_weight[9] = 1
    style_weight = style_weight * 500
    # for i in [10,100,1000]:
        # style_weight = style_weight * 10
        # for weight in [10,100,1000,10000]:
    content_img = imread('./tests/in' + str(13) + '.png')
    style_img = imread('./tests/tar' + str(13) + '.png')
    
    for i in [0, 100, 10000, 100000]:
        run(content_img, style_img, content_weight, style_weight, 0.05, "out" + str(1)+"_", matting_laplacian = True, weight = i, num_steps = 300)
    # for i in range(2, 61):
    #     content_img = imread('./tests/in' + str(i) + '.png')
    #     style_img = imread('./tests/tar' + str(i) + '.png')
    #     run(content_img, style_img, content_weight, style_weight, 0.05, "out" + str(i)+"_", matting_laplacian = True, weight = 50000, num_steps = 300)
    # print(style_weight)
    # print(content_weight)
    # run(content_img, style_img, content_weight, style_weight, 0.05, "out" + str(0)+"_", matting_laplacian = True)
    #different lr
    # for lr in [0.1, 0.001, 0.005, 0.01, 0.05, 0.5]:
    #     style_weight[0]  = 1
    #     style_weight[2]  = 1
    #     style_weight[4]  = 1
    #     style_weight[8]  = 1
    #     style_weight[12] = 1
    #     content_weight[9] = 1
    #     run(content_img, style_img, content_weight, style_weight, lr, "lr=%.3f"%lr)

    # #different alpha
    # for alpha in [0.1, 0.5, 1, 2, 5, 10]:
    #     style_weight[0]  = alpha
    #     style_weight[2]  = alpha
    #     style_weight[4]  = alpha
    #     style_weight[8]  = alpha
    #     style_weight[12] = alpha
    #     content_weight[9] = 1
    #     run(content_img, style_img, content_weight, style_weight, 0.05, "alpha=%.2f"%alpha)

    # #one-hot weight
    # style_weight = torch.zeros(16).cuda()
    # cnt=1
    # for i in [0,2,4,8,12]:
    #     style_weight[i]=10
    #     run(content_img, style_img, content_weight, style_weight, 0.05, "conv_%d"%cnt)
    #     cnt+=1
    #     style_weight[i]=0



