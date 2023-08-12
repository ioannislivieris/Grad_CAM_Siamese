import torch
from torchvision import transforms
from PIL import Image

class Dataset():
    def __init__(self, data:list=None, tfms:transforms=None)->None:
        '''
            Create dataset used in data loader

            Parameters
            ----------
            data: list
                [anchor image, positive/negative image, class]
            tfms: transforms
                transformator

        '''
        self.data = data
        self.tfms = tfms
    
    def __len__(self):
        return len(self.data)     
    
    def __getitem__(self, idx):
        data = self.data[idx]

        image1 = Image.open(data[0]).convert('RGB')
        image2 = Image.open(data[1]).convert('RGB')

        # Apply image transformations
        if self.tfms is not None:
            image1 = self.tfms(image1)
            image2 = self.tfms(image2)
        
        return image1, image2, data[2]  