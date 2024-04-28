import torch
import torchvision.models as models
import torchvision.transforms as transforms
from utils import loadDataset
from utils.constants import *
import xgboost as xgb
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

class TestNeuralNetwork():
    def __init__(self, config):
        self.config = config
        self._vgg = models.vgg16(pretrained=True).to(DEVICE)

    def startTest(self):
        # Initialize dataset
        TestDataset = loadDataset(isTrain=False, modelName=self.config['model_name'])

        # Generate DataLoader
        TestLoader = torch.utils.data.DataLoader(TestDataset, batch_size=BATCH_SIZE, shuffle=False)

        # Load model
        model = None
        if self.config['model_name'] == 'VGG':
            model = self._vgg
        elif self.config['model_name'] == 'XGBoost':
            model = xgb.XGBClassifier()
        else:
            raise ValueError("Please choose either VGG or XGBoost")
        
        # Start test on trained model
        test_accuracy, test_report = self.testModel(model, TestLoader)

        return test_accuracy, test_report

    def testModel(self, model, TestLoader):
        # Evaluate the model
        all_predictions = []
        all_labels = []

        if isinstance(model, torch.nn.Module):  # VGG models
            model.eval()
            with torch.no_grad():
                for images, labels in TestLoader:
                    images, labels = images.to(DEVICE), labels.to(DEVICE)
                    outputs = model(features)
                    _, predicted = torch.max(outputs, 1)
                    all_predictions.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
        else:  # XGBoost models
            for images, labels in TestLoader:
                features = images
                predictions = model.predict(features)
                all_predictions.extend(predictions)
                all_labels.extend(labels.numpy())

        # Calculate accuracy and generate classification report
        accuracy = accuracy_score(all_labels, all_predictions)
        if self.config['model_name'] == 'VGG':
            classes = TestLoader.dataset.classes
        elif self.config['model_name'] == 'XGBoost':
            classes = None  # XGBoost doesn't have class names TODO MISLIM DA XGBOOST 2.0 IMA
        report = classification_report(all_labels, all_predictions, target_names=classes)
        
        return accuracy, report
