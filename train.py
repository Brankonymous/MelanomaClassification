from utils import loadDataset
import torch
from utils.constants import *
import xgboost as xgb
import numpy as np
from test import TestNeuralNetwork
import pickle
import matplotlib.pyplot as plt

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
            loss = torch.nn.CrossEntropyLoss() if self.config['model_name'] == 'VGG' else None
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY) if self.config['model_name'] == 'VGG' else None

            self.cnnTrainLoop(model, loss, optimizer, TrainDataLoader, ValidationDataLoader)

            if self.config['save_plot_train'] or self.config['show_plot_train']:
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
            for i, (images, labels) in enumerate(TrainDataLoader):
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                optimizer.zero_grad()
                prediction = model(images)
                lossValue = loss(prediction, labels)
                lossValue.backward()
                optimizer.step()

                print(f'[Train] Epoch: {epoch}, Batch: {i}, Index: {i*BATCH_SIZE}, Loss: {lossValue.item()}')
            
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
                    print('Early stopping...')
                    break
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

        model.fit(np.concatenate((features_2d, validation_features_2d)), np.concatenate((labels, validation_labels)))
        
        # Best: 0.801709 using {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 200}

    def plotTrainingInfo(self):
        plt.title('Loss by epoch')
        plt.plot(self.lossByEpoch, label='Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        if self.config['save_plot_train']:
            plt.savefig(SAVED_PLOT_PATH + '_' + self.config['model_name'] + '_loss_by_epoch.svg')
        if self.config['show_plot_train']:
            plt.show()
        else:
            plt.close()

        plt.figure(figsize=(10, 8), dpi=150)

        plt.subplot(2, 2, 1)
        plt.title('Accuracy by epoch')
        plt.plot(self.accuracyByEpoch, label='Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')

        plt.subplot(2, 2, 2)
        plt.title('Recall by epoch')
        plt.plot(self.recallByEpoch, label='Recall')
        plt.xlabel('Epoch')
        plt.ylabel('Recall')

        plt.subplot(2, 2, 3)
        plt.title('Precision by epoch')
        plt.plot(self.precisionByEpoch, label='Precision')
        plt.xlabel('Epoch')
        plt.ylabel('Precision')

        plt.subplot(2, 2, 4)
        plt.title('F1 Score by epoch')
        plt.plot(self.f1ScoreByEpoch, label='F1 Score')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')

        plt.tight_layout()
        if self.config['save_plot_train']:
            plt.savefig(SAVED_PLOT_PATH + '_' + self.config['model_name'] + '_metrics_by_epoch.svg')
        if self.config['show_plot_train']:
            plt.show()
        else:
            plt.close()
