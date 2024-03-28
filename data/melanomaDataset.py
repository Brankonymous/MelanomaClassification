from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torchvision.io as io

from utils.constants import *

class MelanomaDataset(Dataset):
    def __init__(self, isTrain=True, transform=None, targetTransform=None):
        self._datasetPath = TRAIN_DATASET_PATH if isTrain else TEST_DATASET_PATH
        self._csvPath = TRAIN_CSV_NAME if isTrain else TEST_CSV_NAME
        
        self._datasetCSV = pd.read_csv(self._csvPath)
        self._transform = transform
        self._targetTransform = targetTransform

    def __len__(self):
        return len(self.dataset_csv)

    def __getitem__(self, idx):
        imagePath = self._datasetPath + '/' + self._datasetCSV.iloc[idx, 0] + '.jpg'
        classLabel = self._datasetCSV.iloc[idx, 1]

        imageTensor = io.read_image(imagePath)

        if self._transform:
            imageTensor = self._transform(imageTensor)

        if self._targetTransform:
            classLabel = self._targetTransform(classLabel)

        return imageTensor, classLabel