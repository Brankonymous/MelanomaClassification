import torch
import torchvision.models as models
import torchvision.transforms as transforms
from utils import loadDataset
from utils.constants import *
import xgboost as xgb
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import pickle 

class TestNeuralNetwork():
    def __init__(self, config):
        self.config = config
        self._vgg = torch.load(SAVED_MODEL_PATH + 'VGG_model.pth')
        self._xgboost = pickle.load(open(SAVED_MODEL_PATH + 'XGBoost_model', 'rb'))

    def startTest(self):
        # Initialize dataset
        TestDataset, _ = loadDataset(isTrain=False, modelName=self.config['model_name'])

        # Generate DataLoader
        TestLoader = torch.utils.data.DataLoader(TestDataset, batch_size=BATCH_SIZE, shuffle=False)

        # Load model
        model = None
        if self.config['model_name'] == 'VGG':
            model = self._vgg
        elif self.config['model_name'] == 'XGBoost':
            model = self._xgboost
        else:
            raise ValueError("Please choose either VGG or XGBoost")
        
        self.testModel(model, TestLoader)

    def testModel(self, model, DataLoader):
        # Evaluate the model
        all_predictions = []
        all_labels = []

        if self.config['model_name'] == 'VGG':
            model.eval()
            with torch.no_grad():
                for images, labels in DataLoader:
                    images, labels = images.to(DEVICE), labels.to(DEVICE)
                    outputs = model(images)
                    prediction = outputs.argmax(1)

                    all_predictions.extend(prediction.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
        elif self.config['model_name'] == 'XGBoost':
            for images, labels in DataLoader:
                features = images
                features_2d = np.array(features).reshape(features.shape[0], -1)
                predictions = model.predict(features_2d)
                all_predictions.extend(predictions)
                all_labels.extend(labels.numpy())

        # Calculate accuracy and generate classification report
        classes = ['Benign', 'Malignant']

        accuracy = accuracy_score(all_labels, all_predictions)
        precision = classification_report(all_labels, all_predictions, target_names=classes, labels=[0, 1], output_dict=True)['weighted avg']['precision']
        recall = classification_report(all_labels, all_predictions, target_names=classes, labels=[0, 1], output_dict=True)['weighted avg']['recall']
        f1_score = classification_report(all_labels, all_predictions, target_names=classes, labels=[0, 1], output_dict=True)['weighted avg']['f1-score']
        report = classification_report(all_labels, all_predictions, target_names=classes, labels=[0, 1])

        print(
            'Accuracy: {:.2f}%'.format(accuracy * 100), '\n', 
            'Precision: {:.2f}%'.format(precision * 100), '\n', 
            'Recall: {:.2f}%'.format(recall * 100), '\n', 
            'F1 Score: {:.2f}%'.format(f1_score * 100), '\n', 
            'Report: ', report
        )

        return f1_score
