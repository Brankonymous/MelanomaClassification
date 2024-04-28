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
        self._vgg = torch.load(SAVED_MODEL_PATH + 'VGG_model.pth')

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

        if self.config['model_name'] == 'VGG':
            model.eval()
            with torch.no_grad():
                for images, labels in TestLoader:
                    images, labels = images.to(DEVICE), labels.to(DEVICE)
                    outputs = model(images)
                    prediction = outputs.argmax(1)

                    all_predictions.extend(prediction.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
        elif self.config['model_name'] == 'XGBoost':
            for images, labels in TestLoader:
                features = images
                predictions = model.predict(features)
                all_predictions.extend(predictions)
                all_labels.extend(labels.numpy())

        # Calculate accuracy and generate classification report
        classes = ['Benign', 'Malignant']
        
        print(all_predictions)
        print(all_labels)

        accuracy = accuracy_score(all_labels, all_predictions)
        report = classification_report(all_labels, all_predictions, target_names=classes)

        print('Accuracy: ', accuracy, '\n', 'Report: ', report)
        
        return accuracy, report
