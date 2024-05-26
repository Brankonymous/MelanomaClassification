from utils import loadDataset
import torch
from utils.constants import *
import xgboost as xgb
import numpy as np
from test import TestNeuralNetwork
import pickle
import matplotlib.pyplot as plt
import torch.nn as nn

class TrainNeuralNetwork():
    def __init__(self, config):
        self.config = config
        self.lossByEpoch = []
        self.accuracyByEpoch = []
        self.recallByEpoch = []
        self.precisionByEpoch = []
        self.f1ScoreByEpoch = []

    def startTrain(self):
        # Initialize dataset
        trainDataset, validationDataset = loadDataset(isTrain=True, modelName=self.config['model_name'])

        # dataloader
        TrainDataLoader = torch.utils.data.DataLoader(trainDataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        ValidationDataLoader = torch.utils.data.DataLoader(validationDataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

        # Load VGG or XGBoost
        if self.config['model_name'] == 'VGG':
            model = torch.hub.load('pytorch/vision:v0.9.0', 'vgg11_bn', pretrained=True).to(DEVICE)
            num_features = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_features, NUM_CLASSES).to(DEVICE)

            weights = torch.tensor([0.19, 0.81], dtype=torch.float).to(DEVICE)
            loss = torch.nn.CrossEntropyLoss(weight=weights).to(DEVICE)
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

            self.cnnTrainLoop(model, loss, optimizer, TrainDataLoader, ValidationDataLoader)

            if self.config['save_plot'] or self.config['show_plot']:
                print('Plotting training info...')
                self.plotTrainingInfo()

            if self.config['save_model']:
                torch.save(model, SAVED_MODEL_PATH + self.config['model_name'] + '_model.pth')
        elif self.config['model_name'] == 'XGBoost':
            model = xgb.XGBClassifier(device=DEVICE_NAME, sample_type='weighted', learning_rate = 0.01, max_depth = 3, n_estimators = 200)
            self.xgbTrainLoop(model, TrainDataLoader, ValidationDataLoader)

            if self.config['save_model']:
                pickle.dump(model, open(SAVED_MODEL_PATH + self.config['model_name'] + "_model", "wb"))
        else:
            raise ValueError("Please choose either VGG or XGBoost")

    def cnnTrainLoop(self, model, loss, optimizer, TrainDataLoader, ValidationDataLoader):
        prev_f1_score, isStop = 0, 3
        for epoch in range(EPOCHS):
            model.train()
            labeler = []
            for i, (images, labels) in enumerate(TrainDataLoader):
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                optimizer.zero_grad()
                prediction = model(images)
                lossValue = loss(prediction, labels)
                lossValue.backward()
                optimizer.step()

                print(f'[Train] Epoch: {epoch}, Batch: {i}, Index: {i*BATCH_SIZE}, Loss: {lossValue.item()}, Learning Rate: {optimizer.param_groups[0]["lr"]}')
                labeler.append(labels)
            labeler = torch.cat(labeler, 0)
            # print(labeler)
            
            # Validate the model
            accuracy, precision, recall, f1_score = TestNeuralNetwork(self.config).testModel(model, ValidationDataLoader)
            self.accuracyByEpoch.append(accuracy)
            self.recallByEpoch.append(recall)
            self.precisionByEpoch.append(precision)
            self.f1ScoreByEpoch.append(f1_score)
            self.lossByEpoch.append(lossValue.item())

            # Early stopping
            if f1_score < prev_f1_score:
                prev_f1_score = 1000
                isStop -= 1
                if isStop == 0:
                    pass
                    # print('Early stopping...')
                    # break
            else:
                prev_f1_score = f1_score


    def xgbTrainLoop(self, model, DataLoader, ValidationDataLoader):
        features, all_labels = [], []
        for i, (images, batch_labels) in enumerate(DataLoader):
            print(f'Batch: {i}, Index: {i*BATCH_SIZE}')
            features.append(images)
            all_labels.append(batch_labels.numpy().tolist())
        
        features = np.concatenate(features, axis=0)
        features_2d = features.reshape(features.shape[0], -1)
        labels = np.concatenate(all_labels, axis=0)

        validation_features, validation_labels = [], []
        for images, batch_labels in ValidationDataLoader:
            validation_features.append(images)
            validation_labels.append(batch_labels.numpy().tolist())

        validation_features = np.concatenate(validation_features, axis=0)
        validation_features_2d = validation_features.reshape(validation_features.shape[0], -1)
        validation_labels = np.concatenate(validation_labels, axis=0)

        # weight
        sample_weights = np.where(np.concatenate((labels, validation_labels)) == 1, 0.81, 0.19)
        model.fit(np.concatenate((features_2d, validation_features_2d)), np.concatenate((labels, validation_labels)), sample_weight=sample_weights)
        
        # Best: 0.801709 using {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 200}

    def plotTrainingInfo(self):
        plt.title('Gubitak po epohi za ' + self.config['model_name'])
        plt.plot(self.lossByEpoch, label='Gubitak')
        plt.xlabel('Epoha')
        plt.ylabel('Gubitak')

        if self.config['save_plot']:
            plt.savefig(SAVED_PLOT_PATH + self.config['model_name'] + '_gubitak_po_epohi.png')
        if self.config['show_plot']:
            plt.show()
        else:
            plt.close()

        plt.figure(figsize=(10, 8), dpi=150)

        plt.figure(figsize=(10, 8), dpi=150)
        plt.suptitle('Metrike po epohi za ' + self.config['model_name'])

        plt.subplot(2, 2, 1)
        plt.title('Tačnost')
        plt.plot(self.accuracyByEpoch, label='Tačnost')
        plt.xlabel('Epoha')
        plt.ylabel('Tačnost')

        plt.subplot(2, 2, 2)
        plt.title('Odziv')
        plt.plot(self.recallByEpoch, label='Odziv')
        plt.xlabel('Epoha')
        plt.ylabel('Odziv')

        plt.subplot(2, 2, 3)
        plt.title('Preciznost')
        plt.plot(self.precisionByEpoch, label='Preciznost')
        plt.xlabel('Epoha')
        plt.ylabel('Preciznost')

        plt.subplot(2, 2, 4)
        plt.title('F1 mera')
        plt.plot(self.f1ScoreByEpoch, label='F1 mera')
        plt.xlabel('Epoha')
        plt.ylabel('F1 mera')

        plt.tight_layout()
        if self.config['save_plot']:
            plt.savefig(SAVED_PLOT_PATH + self.config['model_name'] + '_metrike_po_epohi.png')
        if self.config['show_plot']:
            plt.show()
        else:
            plt.close()
