from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import torch
import torchvision.models as models
from utils.constants import *
from data.customTransforms import *
from random import random
import os 

'''
HAMDataset class is used to load the HAM10000 dataset for testing purposes
'''
class HAMDataset(Dataset):
    def __init__(self, transform=None, targetTransform=None, modelName='VGG'):

        self._datasetPath = HAM10000_DATASET_PATH
        self._csvPath = HAM10000_CSV_NAME
        self._datasetCSV = pd.read_csv(self._csvPath)
        
        self._transform = transform
        self._targetTransform = targetTransform
        self._modelName = modelName

        self.processDataset()

    def __len__(self):
        return len(self._datasetCSV)

    def __getitem__(self, idx):
        imageName = self._datasetCSV.iloc[idx, 1]
        # check if imagePath is in part 1 or part 2
        imagePath1 = self._datasetPath + HAM10000_IMAGES_1_PATH + imageName + '.jpg'
        imagePath2 = self._datasetPath + HAM10000_IMAGES_2_PATH + imageName + '.jpg'

        # check if image is in imagePath1 or imagePath2
        if os.path.exists(imagePath1):
            imagePath = imagePath1
        elif os.path.exists(imagePath2):
            imagePath = imagePath2
        else:
            raise ValueError("Image not found in either path")

        classLabel = self._datasetCSV.iloc[idx, 2]

        image = Image.open(imagePath)

        if self._transform:
            image = self._transform(image)

        if self._targetTransform:
            classLabel = self._targetTransform(classLabel)

        if self._modelName == 'VGG':
            return image, classLabel
        elif self._modelName == 'XGBoost':
            features = torch.flatten(image, 1).numpy()

            return features, classLabel
        
    def processDataset(self):
        pass
