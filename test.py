import torch
import torchvision.models as models
import torchvision.transforms as transforms

class TestNeuralNetwork():
    def __init__(self, config):
        self.config = config
        self.vgg16 = models.vgg16(pretrained=True).to(DEVICE)

    def startTest(self):
        # Initialize dataset
        TestDataset = loadDataset(isTrain=False, model_name=self.config['model_name'])

        # Generate DataLoader
        TestLoader = torch.utils.data.DataLoader(TestDataset, batch_size=BATCH_SIZE, shuffle=False)

        # Load model
        model = None
        if self.config['model_name'] == 'VGG':
            model = self.vgg16
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
                    # Extract features using VGG-16
                    features = self.extract_features(images)
                    outputs = model(features)
                    _, predicted = torch.max(outputs, 1)
                    all_predictions.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
        else:  # XGBoost models
            for images, labels in TestLoader:
                # Placeholder for feature extraction logic
                # You can perform feature extraction here if needed for XGBoost
                # Make sure to adapt this part according to your requirements
                features = images  # Placeholder for features
                predictions = model.predict(features)
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

    def extract_features(self, images):
        # Prepare images for VGG-16 input
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        images = transform(images)
        images = images.to(DEVICE)

        # Extract features using VGG-16
        features = self.vgg16.features(images)
        features = features.view(features.size(0), -1)

        return features
