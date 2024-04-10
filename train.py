from utils import loadDataset
import torch
from utils.constants import *
import warnings
import xgboost as xgb


warnings.filterwarnings('ignore')

class TrainNeuralNetwork():
    def __init__(self, config):
        self.config = config

    def startTrain(self):
        # Initialize dataset
        TrainDataset = loadDataset(isTrain=True)

        # dataloader
        DataLoader = torch.utils.data.DataLoader(TrainDataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

        # Load VGG-16
        model = None
        if self.config['model_name'] == 'VGG':
            model = torch.hub.load('pytorch/vision:v0.9.0', 'vgg16', pretrained=True)
        elif self.config['model_name'] == 'XGBoost':
            model = xgb.XGBClassifier()
        else:
            raise ValueError("Please choose either VGG or XGBoost")

        # Loss function
        loss = torch.nn.CrossEntropyLoss() if self.config['model_name'] == 'VGG' else None

        # Optimizer (not needed for xgboost)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY) if self.config['model_name'] == 'VGG' else None

        self.trainLoop(model, loss, optimizer, DataLoader)

    def trainLoop(self, model, loss, optimizer, DataLoader):
        for epoch in range(EPOCHS):
            model.train()
            for i, (images, labels) in enumerate(DataLoader):
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                #check if optimizer defined (vgg)
                if optimizer: 
                    optimizer.zero_grad()

                #check if model is a Pytorch model(vgg), irelevant for xgboost
                if isinstance(model, torch.nn.Module): 
                    optimizer.zero_grad()
                    outputs = model(images)
                    lossValue = loss(outputs, labels)
                    lossValue.backward()
                    optimizer.step()

                if i % 100 == 0:
                    print(f'Epoch: {epoch}, Batch: {i}, Loss: {lossValue.item()}')