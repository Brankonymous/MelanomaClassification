from utils import loadDataset
import torch
from utils.constants import *
import warnings

warnings.filterwarnings('ignore')

class TrainNeuralNetwork():
    def __init__(self, config):
        self.config = config

    def startTrain(self):
        # Initialize dataset
        TrainDataset = loadDataset(isTrain=True)

        # dataloader
        DataLoader = torch.utils.data.DataLoader(TrainDataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

        # Load VGG
        model = None
        if self.config['model_name'] == 'VGG':
            model = torch.hub.load('pytorch/vision:v0.9.0', 'vgg11_bn', pretrained=True)

        # Loss function
        loss = torch.nn.CrossEntropyLoss()

        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        self.trainLoop(model, loss, optimizer, DataLoader)

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