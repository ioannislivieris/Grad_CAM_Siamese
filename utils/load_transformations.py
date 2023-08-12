from torchvision import transforms

def load_transformations(params:dict=None)->(transforms,transforms):
  '''
    Image transformation for data loader

    Parameters
    ----------
    params: dictionary with parameters

    Returns
    -------
    Tuple with transforms
  '''

  train_tfms = transforms.Compose( [ transforms.Resize( params['image']['size'] ),
                                    # transforms.RandomResizedCrop(224),
                                    # transforms.CenterCrop(224),
                                    # transforms.RandomCrop(48, padding=8, padding_mode='reflect'),                                  
                                    # transforms.ColorJitter(brightness = params['image']['brightness'], 
                                    #                       contrast   = params['image']['contrast'], 
                                    #                       saturation = params['image']['saturation'], 
                                    #                       hue        = params['image']['hue']),
                                    # transforms.RandomHorizontalFlip(p = params['image']['HorizontalFlip']),
                                    # transforms.RandomVerticalFlip(p = params['image']['VerticalFlip']),
                                    # transforms.RandomRotation(degrees = params['image']['RandomRotationDegrees']),
                                    transforms.ToTensor(),
                                    transforms.Normalize(params['image']['normalization']['mean'], params['image']['normalization']['std']),
                                  ] )
  #
  # Testing set
  #
  test_tfms  = transforms.Compose([ transforms.Resize( params['image']['size'] ),
                                    transforms.ToTensor(),
                                    transforms.Normalize(params['image']['normalization']['mean'], params['image']['normalization']['std']),
                                  ] )
  
  return train_tfms, test_tfms