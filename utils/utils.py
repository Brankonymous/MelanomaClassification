from .constants import *
from data.melanomaDataset import MelanomaDataset
from data.customTransforms import LabelToBinary
from torchvision.transforms import ToTensor
from torchvision import transforms

def loadDataset(isTrain=True):
    vggTransform = transforms.Compose([
        
    ])
    targetTransform = transforms.Compose([
        LabelToBinary()
    ])

    dataset = MelanomaDataset(
        isTrain = isTrain,
        transform = vggTransform,
        targetTransform = targetTransform
    )
    
    return dataset