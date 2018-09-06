import cv2
import torch

def preprocess(content_path,style_path,img_h,img_w):

    content=cv2.imread(content_path)
    style=cv2.imread(style_path)

    content=cv2.resize(content,(img_h,img_w))
    style=cv2.resize(style,(img_h,img_w))

    content=content.transpose(2,0,1)
    style=style.transpose(2,0,1)

    content=torch.tensor(content)
    style=torch.tensor(style)

    content=content.unsqueeze(0)
    style=style.unsqueeze(0)
    
    content=torch.FloatTensor(content)
    style=torch.FloatTensor(style)

    content=content/255.0
    style=style/255.0

    return content,style




