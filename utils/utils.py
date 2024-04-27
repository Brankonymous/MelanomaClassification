from .constants import *
from data.melanomaDataset import MelanomaDataset
from data.customTransforms import LabelToBinary
from torchvision.transforms import Normalize, Resize, ToTensor
from torchvision import transforms

def loadDataset(isTrain=True, model_name='VGG'):
    if model_name == 'VGG':
        vggTransform = transforms.Compose([
            ToTensor(),
            Normalize(mean=MEAN_PARAMS, std=STD_PARAMS),
            Resize(RESIZE_PARAMS)
        ])
        targetTransform = transforms.Compose([
            LabelToBinary()
        ])
    
        dataset = MelanomaDataset(
            isTrain = isTrain,
            transform = vggTransform,
            targetTransform = targetTransform
        )
    elif model_name == 'XGBoost':
        #load tabular data, perform preprocessing
        dataset = MelanomaDataset(isTrain=isTrain, transform=None)
    else:
        raise ValueError('Please choose either VGG or XGBoost')
    
    return dataset