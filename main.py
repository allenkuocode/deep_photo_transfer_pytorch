import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import pdb

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np

import copy

from imageio import imread # Image Reading
import matplotlib.pyplot as plt

def show_imgs(content, output, style):
    fig,axes = plt.subplots(1,3,figsize=(12,5),dpi=150)
    imgs=[content, output, style]
    titles=['Content', 'Output', 'Style']
    for i in range(3):
        axes[i].imshow(imgs[i])
        axes[i].set_title(titles[i])
        axes[i].axis("off")
    plt.imsave("lol.png", output)

loader = transforms.Compose([
    transforms.ToTensor()])  # transform it into a torch tensor

def img_to_variable(img):
    img=torch.FloatTensor(img)
    if cuda:
        img=img.cuda()
    img=img.permute(2,0,1)
    return Variable(img).unsqueeze(0)

def variable_to_im(var):
    return var.permute(0,2,3,1).data.cpu().numpy()[0]

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
        return self.loss

class StyleLoss(nn.Module):

    def __init__(self, target, weight):
        super(StyleLoss, self).__init__()
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

class Net(nn.Module):
    def __init__(self, content, style, content_weights, style_weights):
        #pdb.set_trace()
        super(Net, self).__init__()
        vgg=list(models.vgg19(pretrained=True).cuda().features)
        self.vgg=vgg
        self.style_losses=[]
        self.content_losses=[]
        self.x=Variable(content.data.clone().cuda(), requires_grad=True)
        #self.x=nn.Parameter(content.data).cuda()
        self.content_weights = content_weights
        self.style_weights = style_weights

        gram=GramMatrix()
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
                style_loss=StyleLoss(gram(style),self.style_weights[layer_num]* 1000)
                self.style_losses.append(style_loss)
                content_loss=ContentLoss(content, self.content_weights[layer_num])
                self.content_losses.append(style_loss)
                layer_num+=1

    def forward(self):
        vgg=self.vgg
        self.x.data.clamp_(0,1)
        x=self.x
        layer_num=0
        for i in range(len(vgg)):
            x=vgg[i](x)
            if isinstance(vgg[i], nn.Conv2d):
                x = self.style_losses[layer_num](x)
                x = self.content_losses[layer_num](x)
                layer_num+=1

    def backward(self):
        # pdb.set_trace()
        for loss in self.style_losses+self.content_losses:
            loss.backward()
            #loss.backward()




if __name__=='__main__':
    cuda = torch.cuda.is_available()
    content_img = imread('./images/in0.png')
    content_img = content_img[0:-1,:,:]
    #content_img = content_img[0:100,0:100,:] # use subset of image
    style_img = imread('./images/style0.png')
    content_img_var=img_to_variable(content_img)
    #style_img = style_img[0:100,0:100,:] # use subset of image
    style_img_var=img_to_variable(style_img)
    content_weight = torch.zeros(16).cuda()
    content_weight[0] = 1
    content_weight[2] = 1
    content_weight[4] = 1
    content_weight[7] = 1
    content_weight[10] = 1
    style_weight = torch.zeros(16).cuda()
    style_weight[8] = 0
    net=Net(content_img_var, style_img_var, content_weight, style_weight)
    optimizer = optim.Adam([net.x], lr = 0.01)
    if cuda:
        net.cuda()
    step_num=300
    for i in range(step_num):
        optimizer.zero_grad()
        net.forward()
        net.backward()
        optimizer.step()
        if i%50 == 0:
            print("current progress: " + str(i))
    net.x.data.clamp_(0,1)
    show_imgs(content_img, variable_to_im(net.x), style_img)



