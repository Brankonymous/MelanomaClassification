from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import torch
from utils.constants import *
from data.customTransforms import *
import os 
class HAMDataset(Dataset):
    def __init__(self, transform=None, targetTransform=None, modelName='VGG'):
        # Initialize dataset path, CSV path, and read CSV
        self._datasetPath = HAM10000_DATASET_PATH
        self._csvPath = HAM10000_CSV_NAME
        self._datasetCSV = pd.read_csv(self._csvPath)
        
        self._transform = transform
        self._targetTransform = targetTransform
        self._modelName = modelName

    def __len__(self):
        # Return the length of the dataset
        return len(self._datasetCSV)

    def __getitem__(self, idx):
        # Get the image name from the CSV
        imageName = self._datasetCSV.iloc[idx, 1]
        
        # Check if imagePath is in part 1 or part 2
        imagePath1 = self._datasetPath + HAM10000_IMAGES_1_PATH + imageName + '.jpg'
        imagePath2 = self._datasetPath + HAM10000_IMAGES_2_PATH + imageName + '.jpg'

        # Check if image is in imagePath1 or imagePath2
        if os.path.exists(imagePath1):
            imagePath = imagePath1
        elif os.path.exists(imagePath2):
            imagePath = imagePath2
        else:
            raise ValueError("Image not found in either path")

        # Get the class label from the CSV
        classLabel = self._datasetCSV.iloc[idx, 2]

        # Open the image using PIL
        image = Image.open(imagePath)

        # Apply the transform to the image if specified
        if self._transform:
            image = self._transform(image)

        # Apply the target transform to the class label if specified
        if self._targetTransform:
            classLabel = self._targetTransform(classLabel)

        # Return the image and class label based on the model name
        if self._modelName == 'VGG':
            return image, classLabel
        elif self._modelName == 'XGBoost':
            features = torch.flatten(image, 1).numpy()
            return features, classLabel
    
