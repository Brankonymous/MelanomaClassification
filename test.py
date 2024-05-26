import torch
from utils import loadDataset
from utils.constants import *
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import pickle 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

class TestNeuralNetwork():
    def __init__(self, config):
        self.config = config

    def startTest(self):
        # Initialize dataset
        TestDataset, _ = loadDataset(isTrain=False, modelName=self.config['model_name'])

        # Generate DataLoader
        TestLoader = torch.utils.data.DataLoader(TestDataset, batch_size=BATCH_SIZE, shuffle=False)

        # Load model
        model = None
        if self.config['model_name'] == 'VGG':
            model = torch.load(SAVED_MODEL_PATH + 'VGG_model.pth').to(DEVICE)
        elif self.config['model_name'] == 'XGBoost':
            model = pickle.load(open(SAVED_MODEL_PATH + 'XGBoost_model', 'rb'))
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
        # print(all_labels)

        # Calculate accuracy and generate classification report
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = classification_report(all_labels, all_predictions, target_names=CLASS_NAMES, labels=[0, 1], output_dict=True)['weighted avg']['precision']
        recall = classification_report(all_labels, all_predictions, target_names=CLASS_NAMES, labels=[0, 1], output_dict=True)['weighted avg']['recall']
        f1_score = classification_report(all_labels, all_predictions, target_names=CLASS_NAMES, labels=[0, 1], output_dict=True)['weighted avg']['f1-score']
        report = classification_report(all_labels, all_predictions, target_names=CLASS_NAMES, labels=[0, 1])

        print(
            'Accuracy: {:.2f}'.format(accuracy), '\n', 
            'Precision: {:.2f}'.format(precision), '\n', 
            'Recall: {:.2f}'.format(recall), '\n', 
            'F1 Score: {:.2f}'.format(f1_score), '\n', 
            'Report: ', report
        )

        if self.config['save_plot'] or self.config['show_plot']:
            self.plotResults(accuracy, precision, recall, f1_score, all_labels, all_predictions)

        return accuracy, precision, recall, f1_score

    def plotResults(self, accuracy, precision, recall, f1_score, all_labels, all_predictions):
        # Prikaz performansi modela
        fig, ax = plt.subplots(dpi=150)
        ax.bar('Tačnost', accuracy, label='Tačnost')
        ax.bar('Preciznost', precision, label='Preciznost')
        ax.bar('Odziv', recall, label='Odziv')
        ax.bar('F1 Mera', f1_score, label='F1 Mera')
        ax.set_ylim(0, 1)
        ax.set_ylabel('Ocene')
        ax.set_title('Performanse modela ' + self.config['model_name'])
        
        if self.config['save_plot']:
            plt.savefig(SAVED_PLOT_PATH + self.config['model_name'] + '_rezultati.png')
        if self.config['show_plot']:
            plt.show()
        else:
            plt.close()

        # Prikaz matrice konfuzije
        cm = confusion_matrix(all_labels, all_predictions)
        fig, ax = plt.subplots(dpi=150)
        sns.heatmap(cm, annot=True, ax=ax, cmap='Blues', fmt='g', cbar=False)

        ax.set_xlabel('Predviđene klase')
        ax.set_ylabel('Stvarne klase')
        ax.set_title('Matrica konfuzije za model ' + self.config['model_name'])
        ax.xaxis.set_ticklabels(CLASS_NAMES_SERBIAN)
        ax.yaxis.set_ticklabels(CLASS_NAMES_SERBIAN)

        if self.config['save_plot']:
            plt.savefig(SAVED_PLOT_PATH + self.config['model_name'] + '_matrica_konfuzije.png')
        if self.config['show_plot']:
            plt.show()
        else:
            plt.close()
        


