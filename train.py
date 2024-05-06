from utils import loadDataset
import torch
from utils.constants import *
import warnings
import xgboost as xgb
import numpy as np
from test import TestNeuralNetwork
from sklearn.metrics import accuracy_score
import pickle

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

            if self.config['save_model']:
                torch.save(model, SAVED_MODEL_PATH + self.config['model_name'] + '_model.pth')
        elif self.config['model_name'] == 'XGBoost':
            model = xgb.XGBClassifier()
            self.xgbTrainLoop(model, TrainDataLoader, ValidationDataLoader)

            if self.config['save_model']:
                pickle.dump(model, open(SAVED_MODEL_PATH + self.config['model_name'] + "_model", "wb"))
        else:
            raise ValueError("Please choose either VGG or XGBoost")

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


    def xgbTrainLoop(self, model, DataLoader, ValidationDataLoader):
        features, all_labels = [], []
        for i, (images, batch_labels) in enumerate(DataLoader):
            print(f'Batch: {i}, Index: {i*BATCH_SIZE}')
            images = images.to(DEVICE)
            features.append(images)
            all_labels.append(batch_labels.numpy().tolist())
        
        features = np.concatenate(features, axis=0)
        features_2d = features.reshape(features.shape[0], -1)
        labels = np.concatenate(all_labels, axis=0)
        
        model.fit(features_2d, labels)

        # Use the ValidationDataLoader for validation predictions
        validation_features, validation_labels = [], []
        for images, batch_labels in ValidationDataLoader:
            images = images.to(DEVICE)
            validation_features.append(images)
            validation_labels.append(batch_labels.numpy().tolist())

        validation_features = np.concatenate(validation_features, axis=0)
        validation_features_2d = validation_features.reshape(validation_features.shape[0], -1)
        validation_labels = np.concatenate(validation_labels, axis=0)
        validation_predictions = model.predict(validation_features_2d)

        validation_accuracy = accuracy_score(validation_labels, validation_predictions)
        print(f'Validation Accuracy: {validation_accuracy}')