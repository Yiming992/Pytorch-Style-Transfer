    import matplotlib.pyplot as plt
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    content,style=preprocess(content_path,style_path,548,548)

    content=content.to(device)
    style=style.to(device)

    style_trans=Style_Transfer(content,style,548,548,1e-1,1)
    
    alpha=1e+2
    beta=1e+6

    style_trans.to(device)

    epochs=500

    LBFGS=torch.optim.LBFGS([style_trans.target])

    for i in range(epochs):
        def closure():
            style_trans.target.data.clamp_(0,1)
            LBFGS.zero_grad()
            content_loss,style_loss=style_trans()
            total_loss=alpha*content_loss+beta*style_loss
            total_loss.backward()
            return total_loss
        LBFGS.step(closure)
    style_trans.target.data.clamp_(0,1)

    styled_image=style_trans.target
    image=styled_image.squeeze().data.cpu().numpy()
    image=image.transpose(1,2,0)
    plt.imshow(image)
    plt.show()