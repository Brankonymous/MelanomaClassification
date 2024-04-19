class TrainNeuralNetwork():
    def __init__(self, config):
        self.config = config

    def startTrain(self):
        # Initialize dataset
        TrainDataset = loadDataset(isTrain=True, model_name=self.config['model_name'])

        # dataloader
        DataLoader = torch.utils.data.DataLoader(TrainDataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

        # Load VGG-16 or XGBoost
        if self.config['model_name'] == 'VGG':
            model = torch.hub.load('pytorch/vision:v0.9.0', 'vgg16', pretrained=True)
            model.eval()  # Set model to evaluation mode
        elif self.config['model_name'] == 'XGBoost':
            model = xgb.XGBClassifier()
        else:
            raise ValueError("Please choose either VGG or XGBoost")

        # Feature extraction for XGBoost
        if self.config['model_name'] == 'XGBoost':
            features = []
            labels = []
            for images, batch_labels in DataLoader:
                images = images.to(DEVICE)
                # Perform feature extraction using VGG-16 model
                with torch.no_grad():
                    extracted_features = model(images).numpy()
                features.append(extracted_features)
                labels.append(batch_labels.numpy())
            features = np.concatenate(features, axis=0)
            labels = np.concatenate(labels, axis=0)
            # Train XGBoost model with extracted features
            model.fit(features, labels)
            return  # No need to continue training loop for XGBoost

        # Loss function and optimizer for VGG
        loss = torch.nn.CrossEntropyLoss() if self.config['model_name'] == 'VGG' else None
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY) if self.config['model_name'] == 'VGG' else None

        self.trainLoop(model, loss, optimizer, DataLoader)

    def trainLoop(self, model, loss, optimizer, DataLoader):
        for epoch in range(EPOCHS):
            model.train()
            for i, (images, labels) in enumerate(DataLoader):
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                if optimizer:
                    optimizer.zero_grad()

                if isinstance(model, torch.nn.Module):
                    optimizer.zero_grad()
                    outputs = model(images)
                    lossValue = loss(outputs, labels)
                    lossValue.backward()
                    optimizer.step()

                if i % 100 == 0:
                    print(f'Epoch: {epoch}, Batch: {i}, Loss: {lossValue.item()}')
