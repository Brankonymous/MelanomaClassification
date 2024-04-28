from .constants import *
from data.melanomaDataset import MelanomaDataset
from data.customTransforms import LabelToBinary
from torchvision.transforms import Normalize, Resize, ToTensor
from torchvision import transforms

def loadDataset(isTrain=True, modelName='VGG'):
    print(modelName)
    
    vggTransform = transforms.Compose([
        ToTensor(),
        Normalize(mean=MEAN_PARAMS, std=STD_PARAMS),
        Resize(RESIZE_PARAMS)
    ])
    xgboostTransform = transforms.Compose([
        ToTensor(),
        Normalize(mean=MEAN_PARAMS, std=STD_PARAMS),
        Resize(RESIZE_PARAMS)
    ])
    targetTransform = transforms.Compose([
        LabelToBinary()
    ])
    dataset = None

    if modelName == 'VGG':
        dataset = MelanomaDataset(
            isTrain = isTrain,
            transform = vggTransform,
            targetTransform = targetTransform,
            modelName = modelName
        )
    elif modelName == 'XGBoost':
        dataset = MelanomaDataset(
            isTrain = isTrain, 
            transform = xgboostTransform,
            targetTransform = targetTransform,
            modelName = modelName
        )
    else:
        raise ValueError('Please choose either VGG or XGBoost')
    
    return dataset