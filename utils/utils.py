from .constants import *
from data.melanomaDataset import MelanomaDataset
from data.customTransforms import LabelToBinary
from torchvision.transforms import Normalize, Resize, ToTensor
from torchvision import transforms
def loadDataset(isTrain=True):
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
    
    return dataset