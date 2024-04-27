from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models

class MelanomaDataset(Dataset):
    def __init__(self, isTrain=True, transform=None, targetTransform=None):
        self._datasetPath = TRAIN_DATASET_PATH if isTrain else TEST_DATASET_PATH
        self._csvPath = TRAIN_CSV_NAME if isTrain else TEST_CSV_NAME
        
        self._datasetCSV = pd.read_csv(self._csvPath)
        self._transform = transform
        self._targetTransform = targetTransform
        self.vgg16 = models.vgg16(pretrained=True)
        self.vgg16.eval() #no need for further training

    def __len__(self):
        return len(self._datasetCSV)

    def __getitem__(self, idx):
        imagePath = self._datasetPath + '/' + self._datasetCSV.iloc[idx, 0] + '.jpg'
        classLabel = self._datasetCSV.iloc[idx, 1]

        image = Image.open(imagePath)

        if self._transform:
            image = self._transform(image)

        if self._targetTransform:
            classLabel = self._targetTransform(classLabel)
        
        image = transforms.ToTensor()(image)
        image = image.unsqueeze(0)

        # Use VGG-16 to extract features
        with torch.no_grad():
            features = self.vgg16(image)
            features = torch.flatten(features, 1).numpy()

        return features, classLabel
