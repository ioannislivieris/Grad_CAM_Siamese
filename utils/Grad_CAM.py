import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from utils.Siamese import SiameseNetwork

def convert_to_tensor(image:np.ndarray=None, tfms:transforms=None, device:str='cpu')->torch.Tensor:
    '''
        Convert an input image to torch.Tensor

        Parameters
        ----------
        image: input image
        tfms: list of transformations
        device: cpu/cude


        Returns
        -------
        image in torch.Tensor
    '''
    # Apply image transformations
    image = tfms(image).to(device)

    return image


    
def Gram_CAM_heatmap(image:torch.Tensor, model:SiameseNetwork=None, sub_network:int=None,\
image_size:tuple=None, figname:str=None, figsize:tuple=(3,3)):
    '''
        Calculates the heatmap for Gram-CAM procedure

        Parameters
        ----------
        image: requested image
        model: Similarity model (Siamese network)
        sub_network: requested sub_network 1 or 2
        image_size: image size
        figname: figure name (optional)
        fisize: figure size (optional)

        Returns
        -------
        heatmap
    '''

    # pull the gradients out of the model
    gradients = model.get_activations_gradient(sub_network=1)

    # pool the gradients across the channels
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    # get the activations of the last convolutional layer
    activations = model.get_activations(image).detach()

    # weight the channels by corresponding gradients
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]
        
    # average the channels of the activations
    heatmap = torch.mean(activations, dim=1).squeeze()

    # relu on top of the heatmap
    # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
    heatmap = torch.where(heatmap > 0, heatmap, 0)

    # normalize the heatmap
    heatmap /= torch.max(heatmap)

    # Reshape & Convert Tensor to numpy
    heatmap = heatmap.squeeze()
    heatmap = heatmap.detach().cpu().numpy()


    # Resize Heatmap
    heatmap = cv2.resize(heatmap, image_size)
    # Convert to [0,255]
    heatmap = np.uint8(255 * heatmap)


    if figname is not None:
        plt.figure(figsize=figsize);
        fig = plt.imshow(heatmap);
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

        plt.savefig(figname, dpi=300, format='png', 
                bbox_inches='tight', pad_inches=0.1,
                facecolor='auto', edgecolor='auto',
                backend=None, 
            )

    return heatmap