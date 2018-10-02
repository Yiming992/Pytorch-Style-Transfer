import os
from model import Neural_Style
import torch
import argparse
from Preprocess import preprocess 
import matplotlib.pyplot as plt
import cv2
import numpy as np


if __name__=='__main__':

    ap=argparse.ArgumentParser(add_help=False)
    ap.add_argument('-c','--content',required=True)
    ap.add_argument('-s','--style',required=True)
    ap.add_argument('-a','--alpha',default=1e-3)
    ap.add_argument('-b','--beta',default=1.0)
    ap.add_argument('-e','--steps',default=300)
    ap.add_argument('-h','--img_h',default=512)
    ap.add_argument('-w','--img_w',default=512)
    ap.add_argument('-o','--output',default='./outputs/')
    ap.add_argument('-d','--display',default=False)
    ap.add_argument('-n','--name',required=True)


    args=vars(ap.parse_args())

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    content,style=preprocess(args['content'],args['style'],args['img_h'],args['img_w'])
 
    content=content.to(device)
    style=style.to(device)
    neural_style=Neural_Style(content,style)
    neural_style.to(device)

    steps=int(args['steps'])
    LBFGS=torch.optim.LBFGS([neural_style.target])

    alpha=float(args['alpha'])
    beta=float(args['beta'])
    
    i=0
    while i<=steps:
        def closure():
            global i
            neural_style.target.data.clamp_(0,1)
            LBFGS.zero_grad()
            content_loss,style_loss=neural_style()
            total_loss=alpha*content_loss+beta*style_loss
            if i%50==0:
                print('Step:{},Content Loss:{} Style Loss:{} Total Loss:{}'.format(i,content_loss,style_loss,total_loss))
            total_loss.backward()
            i+=1
            return total_loss
        LBFGS.step(closure)
        neural_style.target.data.clamp_(0,1)


    styled_image=neural_style.target
    styled_image=styled_image.squeeze().data.cpu().numpy()
    styled_image=styled_image.transpose(1,2,0)
    styled_image=styled_image*255
    styled_image=styled_image.astype(np.uint8)

    if args['display']:
        plt.imshow(styled_image)
        plt.show()
    
    if not os.path.exists(args['output']):
        os.mkdir(args['output'])
        cv2.imwrite(os.path.join(args['output'],args['name']),styled_image)
    else:
        cv2.imwrite(os.path.join(args['output'],args['name']),styled_image)






















