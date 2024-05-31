from .constants import *
from data.isicDataset import ISICDataset
from data.hamDataset import HAMDataset
from data.customTransforms import IsicToBinary, HamToBinary
from torchvision.transforms import Normalize, Resize, ToTensor, RandomRotation, RandomHorizontalFlip, RandomVerticalFlip
from torchvision import transforms

def loadDataset(isTrain=True, modelName='VGG', datasetName='ISIC'):
    if modelName == 'VGG':
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
    else:
        validationTransform = transforms.Compose([
            ToTensor(),
            Resize(RESIZE_PARAMS)
        ])
        trainAndTestTransform = transforms.Compose([
            ToTensor(),
            Resize(RESIZE_PARAMS),
            RandomRotation(degrees=90),
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5)
        ])

    if datasetName == 'ISIC':
        targetTransform = transforms.Compose([
            IsicToBinary()
        ])
        validationDataset = ISICDataset(
            isTrain = isTrain,
            isVal = True,
            transform = validationTransform,
            targetTransform = targetTransform,
            modelName = modelName
        )
        trainAndTestDataset = ISICDataset(
            isTrain = isTrain,
            transform = trainAndTestTransform,
            targetTransform = targetTransform,
            modelName = modelName
        )

        return trainAndTestDataset, validationDataset
    else:
        targetTransform = transforms.Compose([
            HamToBinary()
        ])

        testDataset = HAMDataset(
            transform = trainAndTestTransform,
            targetTransform = targetTransform,
            modelName = modelName
        )

        return testDataset, None
    
    
    