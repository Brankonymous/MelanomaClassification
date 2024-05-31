from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import torch
from utils.constants import *
from data.customTransforms import *
from random import random

class ISICDataset(Dataset):
    def __init__(self, isTrain=True, isVal=False, transform=None, targetTransform=None, modelName='VGG'):
        # Check if the dataset is for validation and return if it is
        if not isTrain and isVal:
            return

        # Set the dataset path and CSV path based on whether it is for training or testing
        self._datasetPath = TRAIN_DATASET_PATH if isTrain else TEST_DATASET_PATH
        self._csvPath = TRAIN_CSV_NAME if isTrain else TEST_CSV_NAME

        # Read the dataset CSV
        self._datasetCSV = pd.read_csv(self._csvPath)

        # Set the transformation functions and model name
        self._transform = transform
        self._targetTransform = targetTransform
        self._modelName = modelName

        # Process the dataset based on whether it is for validation or training
        if isVal:
            self.processValidationDataset()
        elif isTrain:
            self.processTrainDataset()

    def __len__(self):
        # Return the length of the dataset
        return len(self._datasetCSV)

    def __getitem__(self, idx):
        # Get the image path and class label for the given index
        imagePath = self._datasetPath + '/' + self._datasetCSV.iloc[idx, 0] + '.jpg'
        classLabel = self._datasetCSV.iloc[idx, 1]

        # Open the image using PIL
        image = Image.open(imagePath)

        # Apply the transformation to the image if it is provided
        if self._transform:
            image = self._transform(image)

        # Apply the target transformation to the class label if it is provided
        if self._targetTransform:
            classLabel = self._targetTransform(classLabel)

        # Return the image and class label based on the model name
        if self._modelName == 'VGG':
            return image, classLabel
        elif self._modelName == 'XGBoost':
            features = torch.flatten(image, 1).numpy()
            return features, classLabel

    def processValidationDataset(self):
        # Get the rows with 'benign' and 'malignant' class labels
        benign = self._datasetCSV[self._datasetCSV.iloc[:, 1] == 'benign']
        malignant = self._datasetCSV[self._datasetCSV.iloc[:, 1] == 'malignant']

        # Sample 10% of the rows from each class
        benignValidation = benign.sample(frac=0.1)
        malignantValidation = malignant.sample(frac=0.1)

        # Concatenate the validation samples and save to a CSV file
        validationSet = pd.concat([benignValidation, malignantValidation])
        validationSet.to_csv(VALIDATION_CSV_NAME, index=False)

        # Update the dataset CSV to the validation CSV
        self._datasetCSV = pd.read_csv(VALIDATION_CSV_NAME)

    def processTrainDataset(self):
        # Read the validation CSV
        validationCSV = pd.read_csv(VALIDATION_CSV_NAME)

        # Remove every row that is in the validation set from the dataset CSV
        self._datasetCSV = self._datasetCSV[~self._datasetCSV.isin(validationCSV)].dropna()
