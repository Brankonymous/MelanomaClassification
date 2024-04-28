from utils import loadDataset
import torch
from utils.constants import *
import warnings
import xgboost as xgb
import numpy as np

class TrainNeuralNetwork():
    def __init__(self, config):
        self.config = config

    def startTrain(self):
        # Initialize dataset
        TrainDataset = loadDataset(isTrain=True, modelName=self.config['model_name'])

        # dataloader
        DataLoader = torch.utils.data.DataLoader(TrainDataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

        # Load VGG or XGBoost
        if self.config['model_name'] == 'VGG':
            model = torch.hub.load('pytorch/vision:v0.9.0', 'vgg11_bn', pretrained=True)
            # model.eval()  # Set model to evaluation mode
        elif self.config['model_name'] == 'XGBoost':
            model = xgb.XGBClassifier()
        else:
            raise ValueError("Please choose either VGG or XGBoost")

        # Feature extraction for XGBoost
        if self.config['model_name'] == 'XGBoost':
            features, labels = [], []
            for i, (images, labels) in enumerate(DataLoader):
                print(f'Batch: {i}, Index: {i*BATCH_SIZE}')
                images = images.to(DEVICE)
                features.append(images)
                labels.append(labels.numpy())
            
            features = np.concatenate(features, axis=0)
            labels = np.concatenate(labels, axis=0)
            model.fit(features, labels)
            return 

        # Loss function and optimizer for VGG
        loss = torch.nn.CrossEntropyLoss() if self.config['model_name'] == 'VGG' else None
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY) if self.config['model_name'] == 'VGG' else None

        self.trainLoop(model, loss, optimizer, DataLoader)

        if self.config['save_model']:
            torch.save(model, SAVED_MODEL_PATH + self.config['model_name'] + '_model.pth')

    def trainLoop(self, model, loss, optimizer, DataLoader):
        for epoch in range(2):
            currentAccuracy = 0
            model.train()
            for i, (images, labels) in enumerate(DataLoader):
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                optimizer.zero_grad()
                prediction = model(images)
                lossValue = loss(prediction, labels)
                lossValue.backward()
                optimizer.step()

                currentAccuracy += (prediction.argmax(1) == labels).sum().item()

                print(f'Epoch: {epoch}, Batch: {i}, Index: {i*BATCH_SIZE}, Loss: {lossValue.item()}')
            currentAccuracy = (currentAccuracy * 100) / len(DataLoader.dataset)

            print(f'Epoch: {epoch}, Accuracy: {currentAccuracy}')
