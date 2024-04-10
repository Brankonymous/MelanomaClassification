from utils import loadDataset
import torch
from utils.constants import *
from sklearn.metrics import classification_report, accuracy_score
import warnings
import xgboost as xgb

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
        model = None
        if self.config['model_name'] == 'VGG':
            model = torch.hub.load('pytorch/vision:v0.9.0', 'vgg16', pretrained=True)
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

        if isinstance(model, torch.nn.Module): #VGG models
            model.eval()
            with torch.no_grad():
                for images, labels in TestLoader:
                    images, labels = images.to(DEVICE), labels.to(DEVICE)
                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)
                    all_predictions.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
        else: #XGBoost models
            for images, labels in TestLoader:
                predictions = model.predict(images)
                all_predictions.extend(predictions)
                all_labels.extend(labels.numpy())

        # Calculate accuracy and generate classification report
        accuracy = accuracy_score(all_labels, all_predictions)
        if self.config['model_name'] == 'VGG':
            classes = TestLoader.dataset.classes
        elif self.config['model_name'] == 'XGBoost':
            classes = None  # XGBoost doesn't have class names
        report = classification_report(all_labels, all_predictions, target_names=classes)
        
        return accuracy, report
