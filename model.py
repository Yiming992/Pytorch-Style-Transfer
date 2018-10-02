import torch 
import torch.nn as nn
from torchvision import models 
import numpy as np


class Neural_Style(nn.Module):

    def __init__(self,content,style):

        super(Neural_Style,self).__init__()
        self.content=content
        self.style=style
        self.content_layer=[21]
        self.style_layer=[0,5,10,19,28]
        self.content_rep=[]
        self.style_rep=[]

        vgg19=models.vgg19(pretrained=True)
        for param in vgg19.parameters():
            param.requires_grad_(False)
        self.vgg19=list(vgg19.children())[0]
        self.target=nn.Parameter(self.content.clone())


    def forward(self):
        N_content=[]
        N_style=[]
        if not (len(self.content_rep)!=0 and len(self.style_rep)==5):
            x=self.content
            y=self.style 
            z=self.target
            for index,model in enumerate(self.vgg19):
                x=model(x)
                y=model(y)
                z=model(z)
                if index in self.content_layer:
                    self.content_rep.append(x)
                    N_content.append(z)
                elif index in self.style_layer:
                    self.style_rep.append(y)
                    N_style.append(z)
            content_loss=self.content_loss(self.content_rep,N_content)
            style_loss=self.style_loss(self.style_rep,N_style)
        else:
            z=self.target
            for index,model in enumerate(self.vgg19):
                z=model(z)
                if index in self.content_layer:
                    N_content.append(z)
                elif index in self.style_layer:
                    N_style.append(z)
            content_loss=self.content_loss(self.content_rep,N_content)
            style_loss=self.style_loss(self.style_rep,N_style)
        return content_loss,style_loss 


    def content_loss(self,content,N_content):
        content=content[0]
        N_content=N_content[0]
        content=content.view(1,content.size(1),content.size(2)*content.size(3))
        N_content=N_content.view(1,N_content.size(1),N_content.size(2)*N_content.size(3))
        N=content.size(1)
        M=content.size(2)
        content_loss=torch.sum(torch.pow(content-N_content,2)/(N*M))/2
        return content_loss


    def style_loss(self,style,N_style):
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
            style_loss+=(torch.sum(torch.pow(i-j,2))/((2*N*M)**2))/5
        return style_loss
