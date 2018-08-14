import torch 
import torch.nn as nn
from torchvision import models 
import numpy as np
from preprocess
 

class Style_Transfer(nn.Module):

    def __init__(self,content,style,img_h,img_w,alpha,beta):

        super(CNN,self).__init__()

        self.content=content
        self.style=style
        self.h=img_h
        self.w=img_w
        self.alpha=alpha
        self.beta=beta
        self.content_layer=[26]
        self.style_layer=[1,6,11,20,29]

        vgg19=models.vgg19(pretrained=True)
        for param in vgg19.parameters():
            param.requires_grad_(False)

        self.vgg19=nn.Sequential(list(vgg19.children())[0])

        self.target=nn.Parameter(torch.rand(1,3,img_h,img_w))


    def forward(self):
        style=[]
        N_style=[]
        x=self.content
        y=self.style
        z=self.target
        for index,model in self.vgg19:
            x=model(x)
            y=model(y)
            z=model(z)
            if index in self.content_layer:
                content=x
                N_content=z
            elif index in self.style_layer:
                style.append(y)
                N_style.append(z)
        
        content_loss=self.content_loss(content,N_content)
        style_loss=self.style_loss(style,N_style)

        return content_loss,style_loss 


    def content_loss(self,content,N_content):
        content=content.view(1,content.size(1),content.size(2)*content.size(3))
        N_content=N_content.view(1,N_content.size(1),N_content.size(2)*N_content.size(3))
        content_loss=torch.sum(torch.pow(content-N_content,2))/2
        return content_loss


        

    def style_loss(self,style,N_style,w):
        style_loss=0.0
        for i,j in zip(style,N_style):
            i=i.view(1,i.size(1),i.size(2)*i.size(3))
            j=j.view(1,j.size(1),j.size(2)*j.size(3))
            N=i.size(1)
            M=i.size(2)
            i=i.squeeze()
            j=j.squeeze()
            i=torch.matmul(i,i.transpose(1,0))
            j=torch.matmul(j,j.transpose(1,0))
            style_loss+=torch.sum(torch.pow(i-j,2)/((2*N*M)**2))/5
        return style_loss
