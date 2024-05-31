from .constants import *
from data.isicDataset import ISICDataset
from data.hamDataset import HAMDataset
from data.customTransforms import IsicToBinary, HamToBinary
from torchvision.transforms import Normalize, Resize, ToTensor, RandomRotation, RandomHorizontalFlip, RandomVerticalFlip
from torchvision import transforms

def loadDataset(isTrain=True, modelName='VGG', datasetName='ISIC'):
    # Define transforms based on the selected model
    if modelName == 'VGG':
        # Transforms for validation dataset
        validationTransform = transforms.Compose([
            ToTensor(),
            Normalize(mean=MEAN_PARAMS, std=STD_PARAMS),
            Resize(RESIZE_PARAMS)
        ])
        # Transforms for train and test datasets
        trainAndTestTransform = transforms.Compose([
            ToTensor(),
            Normalize(mean=MEAN_PARAMS, std=STD_PARAMS),
            Resize(RESIZE_PARAMS),
            RandomRotation(degrees=90),
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5)
        ])
    elif modelName == 'XGBoost':
        # Transforms for validation dataset
        validationTransform = transforms.Compose([
            ToTensor(),
            Resize(RESIZE_PARAMS)
        ])
        # Transforms for train and test datasets
        trainAndTestTransform = transforms.Compose([
            ToTensor(),
            Resize(RESIZE_PARAMS),
            RandomRotation(degrees=90),
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5)
        ])

    # Load dataset based on the selected dataset name
    if datasetName == 'ISIC':
        # Define target transform for ISIC dataset
        targetTransform = transforms.Compose([
            IsicToBinary()
        ])
        # Create validation dataset
        validationDataset = ISICDataset(
            isTrain=isTrain,
            isVal=True,
            transform=validationTransform,
            targetTransform=targetTransform,
            modelName=modelName
        )
        # Create train and test dataset
        trainAndTestDataset = ISICDataset(
            isTrain=isTrain,
            transform=trainAndTestTransform,
            targetTransform=targetTransform,
            modelName=modelName
        )

        return trainAndTestDataset, validationDataset
    elif datasetName == 'HAM':
        # Define target transform for HAM dataset
        targetTransform = transforms.Compose([
            HamToBinary()
        ])
        # Create test dataset
        testDataset = HAMDataset(
            transform=trainAndTestTransform,
            targetTransform=targetTransform,
            modelName=modelName
        )

        return testDataset, None