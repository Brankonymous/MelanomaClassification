from utils import loadDataset

class TrainNeuralNetwork():
    def __init__(self, config):
        self.config = config

    def startTrain(self, validationFold):
        # Initialize dataset
        trainDataset = loadDataset(isTrain=True)

        # Generate DataLoader

        # Load model

        # Train and validate neural network

        pass
