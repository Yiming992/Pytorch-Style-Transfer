import cv2
import torch
import torch.nn as nn

class Normalization(nn.Module):


    def __init__(self,device=False):

        super(Normalization,self).__init__()
        if device:
            self.mean=torch.tensor([0.485,0.456,0.406]).to(device)
            self.std=torch.tensor([0.229,0.224,0.225]).to(device)
        else:
            self.mean=torch.tensor([0.485,0.456,0.406])
            self.std=torch.tensor([0.229,0.224,0.225])


    def forward(self,img):

        self.mean=self.mean.view(1,3,1,1)
        self.std=self.std.view(1,3,1,1)

        return (img-self.mean)/self.std

    def reverse(self,img):
        self.mean=self.mean.view(1,3,1,1)
        self.std=self.std.view(1,3,1,1)

        return (img+self.mean)*self.std


def preprocess(content_path,style_path,img_h,img_w):

    content=cv2.imread(content_path)
    style=cv2.imread(style_path)
    content=content[:,::-1]
    style=style[:,::-1]
    content=cv2.resize(content,(img_h,img_w))
    style=cv2.resize(style,(img_h,img_w))
    content=content.transpose(2,0,1)
    style=style.transpose(2,0,1)
    content=torch.tensor(content)
    style=torch.tensor(style)
    content=content.unsqueeze(0)
    style=style.unsqueeze(0)    
    content=content.float()
    style=style.float()    
    content=content/255.0
    style=style/255.0
    # normalize=Normalization()
    # content,style=normalize(content),normalize(style)
    return content,style




