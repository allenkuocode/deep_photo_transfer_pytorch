import torch
import torch.nn as nn
from torch.autograd import Variable

if __name__=='__main__':
    x=x=Variable(torch.randn((2,3)), requires_grad=True)
    y=2*x
    print(y.backward(torch.Tensor([2,3])))