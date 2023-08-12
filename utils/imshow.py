import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

def imshow(image:torch.Tensor=None, heatmap:np.ndarray=None, scale:float=0.4, mean:tuple=(0, 0, 0), std:tuple=(1, 1, 1), figsize:tuple=(10,10), figname:str=None):
    '''
        Show and save image
        
        Parameters
        ----------
        image: image as torch.Tensor
        heatmap: Grad-CAM heatmap (optional)
        scale: Scaling parameters for merging Grad-CAM heatmap with image
    '''
    # Convert image to NumPy
    npimg = image.squeeze(0).cpu().numpy()

    # Image normalization
    npimg = npimg * np.array(std)[:,None,None] + np.array(mean)[:,None,None]  # Unnormalize

    # Reshape
    npimg = np.transpose(npimg, (1, 2, 0))

    if heatmap is not None:
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET) / 255.0
        npimg = heatmap * scale + npimg
        

    # Show image
    plt.figure(figsize=figsize)
    fig = plt.imshow(npimg)
    if figname is not None:
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

        plt.savefig(figname, dpi=300, 
                bbox_inches='tight', pad_inches=0.1,
                facecolor='auto', edgecolor='auto',
                backend=None, 
            )        