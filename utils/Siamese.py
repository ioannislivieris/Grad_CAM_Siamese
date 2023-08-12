import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from   torch.autograd   import Variable 
from typing import Any, Dict, Sequence, Union
TorchData = Union[Dict[str, torch.Tensor], Sequence[torch.Tensor], torch.Tensor]


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, label, distance):
        '''
        Contrastive loss

        Parameters
        ----------
        label: true label
        distance: distance between instances

        Returns
        -------
        Loss value
        '''        
        loss_contrastive = torch.mean( (1.0-label) * torch.pow(distance, 2) + (label) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2))

        return loss_contrastive
        
        

class Softplus(nn.Module):
    def __init__(self):
        super(Softplus, self).__init__()
    def forward(self, x):
        return torch.where(x < 1.0, x, 1.0)
    

class Flatten(nn.Module):
    def forward(self, *args: TorchData, **kwargs: Any) -> torch.Tensor:
        assert len(args) == 1
        x = args[0]
        assert isinstance(x, torch.Tensor)
        return x.view(x.size(0), -1)
    

class SiameseNetwork(nn.Module):
    def __init__(self, image_size=(3, 224, 224), backbone_model:str='MobileNet', pretrained:bool=True):
        super(SiameseNetwork, self).__init__()
        
        self.backbone_model = backbone_model
        self.distance = F.pairwise_distance
        self.loss_fnc = ContrastiveLoss()
        # Placeholder for the gradients
        self.gradients_1 = None
        self.gradients_2 = None





        # Backbone model and pooling layer
        # -------------------------------------------------------------------------------------------
        if self.backbone_model == 'MobileNet':
            # Siamese network backbone: MobileNet_V2
            self.backbone = torchvision.models.mobilenet_v2(pretrained=pretrained).features
            # Pooling
            self.pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        elif self.backbone_model == 'ResNet50':
            # Siamese network backbone: ResNet50
            model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
            self.backbone = nn.Sequential(*list(model.children())[:-2])
            # Pooling
            self.pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        else:
            raise Exception('Not known pre-trained model')



        # Dense-layer
        # -------------------------------------------------------------------------------------------
        n_size = self._get_conv_output( image_size )
        self.fc = nn.Sequential(
            nn.Linear(in_features=n_size, out_features=256),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=256, out_features=2),  
        )    



    # generate input sample and forward to get shape
    def _get_conv_output(self, shape):
        bs = 1
        input = Variable( torch.rand(bs, *shape) )
        output_feat = self._forward_features( input )
        return output_feat.data.view(bs, -1).size(1)

    def _forward_features(self, x):
        output = self.backbone(x)
        return self.pooling(output)

    # method for the activation extraction
    def get_activations(self, x):
        return self.backbone(x)
    
    # hook for the gradients of the activations
    def activations_hook_image1(self, grad):
        self.gradients_1 = grad
    def activations_hook_image2(self, grad):
        self.gradients_2 = grad

    def get_activations_gradient(self, sub_network=1):
        if sub_network == 1:
            return self.gradients_1
        else:
            return self.gradients_2


    def forward(self, image1, image2, label=None):
        # Sub-Network 1
        # ----------------------------------------------------------
        embeddings1 = self.backbone(image1)
        # register the hook
        _ = embeddings1.register_hook(self.activations_hook_image1)   
        # apply the remaining pooling
        embeddings1 = self.pooling(embeddings1)
        embeddings1 = Flatten()(embeddings1)
        embeddings1 = self.fc(embeddings1)


        # Sub-Network 2
        # ----------------------------------------------------------
        embeddings2 = self.backbone(image2)        
        # register the hook
        _ = embeddings2.register_hook(self.activations_hook_image2)
        # apply the remaining pooling
        embeddings2 = self.pooling(embeddings2)
        embeddings2 = Flatten()(embeddings2)
        embeddings2 = self.fc(embeddings2)   

        distance = self.distance(embeddings1, embeddings2)
        distance = Softplus()(distance)
        if label is not None:
            loss = self.loss_fnc(label, distance.double())
            return distance, loss
        else:
            return distance
        










