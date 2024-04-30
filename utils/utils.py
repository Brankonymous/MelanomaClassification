from .constants import *
from data.melanomaDataset import MelanomaDataset
from data.customTransforms import LabelToBinary
from torchvision.transforms import Normalize, Resize, ToTensor, RandomRotation, RandomHorizontalFlip, RandomVerticalFlip
from torchvision import transforms

def loadDataset(isTrain=True, modelName='VGG'):
    validationTransform = transforms.Compose([
        ToTensor(),
        Normalize(mean=MEAN_PARAMS, std=STD_PARAMS),
        Resize(RESIZE_PARAMS)
    ])
    trainAndTestTransform = transforms.Compose([
        ToTensor(),
        Normalize(mean=MEAN_PARAMS, std=STD_PARAMS),
        Resize(RESIZE_PARAMS),
        RandomRotation(degrees=90),
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5)
    ])

    targetTransform = transforms.Compose([
        LabelToBinary()
    ])

    if modelName == 'VGG':
        validationDataset = MelanomaDataset(
            isTrain = isTrain,
            isVal = True,
            transform = validationTransform,
            targetTransform = targetTransform,
            modelName = modelName
        )
        trainAndTestDataset = MelanomaDataset(
            isTrain = isTrain,
            transform = trainAndTestTransform,
            targetTransform = targetTransform,
            modelName = modelName
        )
    elif modelName == 'XGBoost':
        validationDataset = MelanomaDataset(
            isTrain = isTrain,
            isVal = True,
            transform = validationTransform,
            targetTransform = targetTransform,
            modelName = modelName
        )
        trainAndTestDataset = MelanomaDataset(
            isTrain = isTrain, 
            transform = trainAndTestTransform,
            targetTransform = targetTransform,
            modelName = modelName
        )
    else:
        raise ValueError('Please choose either VGG or XGBoost')
    
    return trainAndTestDataset, validationDataset