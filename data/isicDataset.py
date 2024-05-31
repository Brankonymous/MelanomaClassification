from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import torch
import torchvision.models as models
from utils.constants import *
from data.customTransforms import *
from random import random

class ISICDataset(Dataset):
    def __init__(self, isTrain=True, isVal=False, transform=None, targetTransform=None, modelName='VGG'):
        if not isTrain and isVal:
            return

        self._datasetPath = TRAIN_DATASET_PATH if isTrain else TEST_DATASET_PATH
        self._csvPath = TRAIN_CSV_NAME if isTrain else TEST_CSV_NAME
        self._datasetCSV = pd.read_csv(self._csvPath)
        
        self._transform = transform
        self._targetTransform = targetTransform
        self._modelName = modelName

        if isVal:
            self.processValidationDataset()
        elif isTrain:
            self.processTrainDataset()

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

        if self._modelName == 'VGG':
            return image, classLabel
        elif self._modelName == 'XGBoost':
            features = torch.flatten(image, 1).numpy()

            return features, classLabel
        
    def processValidationDataset(self):
        # Extract 10% of the dataset for validation for each class

        benign = self._datasetCSV[self._datasetCSV.iloc[:, 1] == 'benign']
        malignant = self._datasetCSV[self._datasetCSV.iloc[:, 1] == 'malignant']

        benignValidation = benign.sample(frac=0.1)
        malignantValidation = malignant.sample(frac=0.1)

        validationSet = pd.concat([benignValidation, malignantValidation])
        validationSet.to_csv(VALIDATION_CSV_NAME, index=False)
        
        self._datasetCSV = pd.read_csv(VALIDATION_CSV_NAME)

    def processTrainDataset(self):
        # Extract 10% of the dataset for validation for each class

        validationCSV = pd.read_csv(VALIDATION_CSV_NAME)

        # Remove every row that is in the validation set
        self._datasetCSV = self._datasetCSV[~self._datasetCSV.isin(validationCSV)].dropna()
