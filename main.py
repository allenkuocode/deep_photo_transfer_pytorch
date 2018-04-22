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

unloader = transforms.ToPILImage()

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

vgg=None
class Net(nn.Module):
    def __init__(self, content, style, content_weights, style_weights):
        #pdb.set_trace()
        super(Net, self).__init__()

        self.vgg=vgg
        self.style_losses=[]
        self.content_losses=[]
        self.x=Variable(content.data.clone().cuda(), requires_grad=True)
        #self.x=Variable(torch.rand(content.data.shape).cuda(), requires_grad=True)
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
                style_loss=StyleLoss(gram(style),self.style_weights[layer_num]).cuda()
                self.style_losses.append(style_loss)
                content_loss=ContentLoss(content, self.content_weights[layer_num]).cuda()
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
        loss_score=0
        for loss in self.style_losses+self.content_losses:
            loss_score+=loss.backward()
        return loss_score
            #loss.backward()


def image_loader(image_name):
    image = Image.open(image_name)
    image = Variable(loader(image))
    # fake batch dimension required to fit network's input dimensions
    image = image.unsqueeze(0)
    return image

def run(content_img, style_img, content_weight, style_weight, lr, fname):

    content_img_var=img_to_variable(content_img)
    style_img_var=img_to_variable(style_img)

    #conv1_1, conv_2_1, conv_3_1, conv_4_1,conv_5_1
    net=Net(content_img_var, style_img_var, content_weight, style_weight)
    optimizer = optim.Adam([net.x], lr=lr)
    if cuda:
        net.cuda()
    step_num=300
    for i in range(step_num):
        optimizer.zero_grad()
        net.forward()
        loss=net.backward()
        optimizer.step()
        #if i%50 == 0:
        print("current progress: " + str(i))
    net.x.data.clamp_(0,1)
    #show_imgs(content_img, variable_to_im(net.x), style_img)
    plt.imsave('output/'+fname+'.jpg', variable_to_im(net.x))

if __name__=='__main__':
    os.system('rm -rf output')
    os.mkdir('output')
    cuda = torch.cuda.is_available()
    style_weight = torch.zeros(16).cuda()
    content_weight = torch.zeros(16).cuda()
    content_img = imread('./images/in1.jpg')
    style_img = imread('./images/style1.jpg')


    vgg=list(models.vgg19(pretrained=True).features.cuda())

    #different lr
    for lr in [0.1, 0.001, 0.005, 0.01, 0.05, 0.5]:
        style_weight[0]  = 1
        style_weight[2]  = 1
        style_weight[4]  = 1
        style_weight[8]  = 1
        style_weight[12] = 1
        content_weight[9] = 1
        run(content_img, style_img, content_weight, style_weight, lr, "lr=%.3f"%lr)

    #different alpha
    for alpha in [0.1, 0.5, 1, 2, 5, 10]:
        style_weight[0]  = alpha
        style_weight[2]  = alpha
        style_weight[4]  = alpha
        style_weight[8]  = alpha
        style_weight[12] = alpha
        content_weight[9] = 1
        run(content_img, style_img, content_weight, style_weight, 0.05, "alpha=%.2f"%alpha)

    #one-hot weight
    style_weight = torch.zeros(16).cuda()
    cnt=1
    for i in [0,2,4,8,12]:
        style_weight[i]=10
        run(content_img, style_img, content_weight, style_weight, 0.05, "conv_%d"%cnt)
        cnt+=1
        style_weight[i]=0



