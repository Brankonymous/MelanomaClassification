import argparse
from train import TrainNeuralNetwork
from test import TestNeuralNetwork
from utils.constants import *
import datetime

def train(config):
    start = datetime.datetime.now()
    print('[Train] Starting...') 
    for val_fold in range(1, K_FOLD+1):
        print(f'--------- Validation fold {val_fold} ---------')
        
        trainNeuralNet = TrainNeuralNetwork(config=config)
        trainNeuralNet.startTrain(val_fold)
    print('[Train] Time taken: ', datetime.datetime.now() - start)

def test(config):
    testNeuralNet = TestNeuralNetwork(config=config)
    for val_fold in range(1, K_FOLD+1):
        testNeuralNet.startTest(val_fold)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Common params
    parser.add_argument('--type', choices=[m.name for m in ModelType], type=str, help='Input TRAIN, TEST or TRAIN_AND_TEST for type of classification', default=ModelType.TRAIN.name)
    parser.add_argument('--model_name', choices=[m.name for m in SupportedModels], type=str, help='Neural network (model) to use', default=SupportedModels.VGG.name)
    parser.add_argument('--save_model', help='Save model during training', default=True)
    
    # Wrapping configuration into a dictionary
    args = parser.parse_args()
    config = dict()
    for arg in vars(args):
        config[arg] = getattr(args, arg)
    
    if config['type'] == 'TRAIN' or config['type'] == 'TRAIN_AND_TEST':
        train(config)
    if config['type'] == 'TEST' or config['type'] == 'TRAIN_AND_TEST':
        test(config)
    

    