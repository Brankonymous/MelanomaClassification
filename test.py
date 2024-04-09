from utils import loadDataset
import torch
from utils.constants import *
from sklearn.metrics import classification_report, accuracy_score
import warnings

warnings.filterwarnings('ignore')

class TestNeuralNetwork():
    def __init__(self, config):
        self.config = config

    def startTest(self):
        # Initialize dataset
        TestDataset = loadDataset(isTrain=False)

        # Generate DataLoader
        TestLoader = torch.utils.data.DataLoader(TestDataset, batch_size=BATCH_SIZE, shuffle=False)

        # Load model
        model = torch.hub.load('pytorch/vision:v0.9.0', 'vgg16', pretrained=True)
        
        # Start test on trained model
        test_accuracy, test_report = self.testModel(model, TestLoader)

        return test_accuracy, test_report

    def testModel(self, model, TestLoader):
        # Evaluate the model
        all_predictions = []
        all_labels = []

        model.eval()
        with torch.no_grad():
            for images, labels in TestLoader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate accuracy and generate classification report
        accuracy = accuracy_score(all_labels, all_predictions)
        report = classification_report(all_labels, all_predictions, target_names=test_dataset.classes)

        return accuracy, report
