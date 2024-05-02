from utils import loadDataset
import torch
from utils.constants import *
import warnings
import xgboost as xgb
import numpy as np
from test import TestNeuralNetwork

class TrainNeuralNetwork():
    def __init__(self, config):
        self.config = config

    def startTrain(self):
        # Initialize dataset
        trainDataset, validationDataset = loadDataset(isTrain=True, modelName=self.config['model_name'])

        # dataloader
        TrainDataLoader = torch.utils.data.DataLoader(trainDataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        ValidationDataLoader = torch.utils.data.DataLoader(validationDataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

        # Load VGG or XGBoost
        if self.config['model_name'] == 'VGG':
            model = torch.hub.load('pytorch/vision:v0.9.0', 'vgg11_bn', pretrained=True)
            loss = torch.nn.CrossEntropyLoss() if self.config['model_name'] == 'VGG' else None
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY) if self.config['model_name'] == 'VGG' else None

            self.cnnTrainLoop(model, loss, optimizer, TrainDataLoader, ValidationDataLoader)
        elif self.config['model_name'] == 'XGBoost':
            model = xgb.XGBClassifier()
            self.xgbTrainLoop(model, TrainDataLoader)
        else:
            raise ValueError("Please choose either VGG or XGBoost")

        if self.config['save_model']:
            torch.save(model, SAVED_MODEL_PATH + self.config['model_name'] + '_model.pth')

    def cnnTrainLoop(self, model, loss, optimizer, TrainDataLoader, ValidationDataLoader):
        prev_f1_score, isStop = 0, 2
        for epoch in range(EPOCHS):
            model.train()
            for i, (images, labels) in enumerate(TrainDataLoader):
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                optimizer.zero_grad()
                prediction = model(images)
                lossValue = loss(prediction, labels)
                lossValue.backward()
                optimizer.step()

                print(f'[Train] Epoch: {epoch}, Batch: {i}, Index: {i*BATCH_SIZE}, Loss: {lossValue.item()}')
            
            # Validate the model
            f1_score = TestNeuralNetwork(self.config).testModel(model, ValidationDataLoader)

            # Early stopping
            if f1_score < prev_f1_score:
                prev_f1_score = 1000
                isStop -= 1
                if isStop == 0:
                    print('Early stopping')
                    break
            else:
                prev_f1_score = f1_score


    def xgbTrainLoop(self, model, DataLoader):
        features, labels = [], []
        for i, (images, labels) in enumerate(DataLoader):
            print(f'Batch: {i}, Index: {i*BATCH_SIZE}')
            images = images.to(DEVICE)
            features.append(images)
            labels.append(labels.numpy())
        
        features = np.concatenate(features, axis=0)
        labels = np.concatenate(labels, axis=0)
        model.fit(features, labels)