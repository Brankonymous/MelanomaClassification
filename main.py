import argparse
from train import TrainNeuralNetwork
from test import TestNeuralNetwork
from utils.constants import *
import datetime
import warnings

def train(config):
    start = datetime.datetime.now()
    print('[Train] Starting...') 
        
    trainNeuralNet = TrainNeuralNetwork(config=config)
    trainNeuralNet.startTrain()
    print('[Train] Time taken: ', datetime.datetime.now() - start)

def test(config):
    start = datetime.datetime.now()
    print('[Test] Starting...') 
        
    testNeuralNet = TestNeuralNetwork(config=config)
    testNeuralNet.startTest()
    print('[Test] Time taken: ', datetime.datetime.now() - start)

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser()

    # Common params
    parser.add_argument('--type', choices=[m.name for m in ModelType], type=str, help='Input TRAIN, TEST or TRAIN_AND_TEST for type of classification', default=ModelType.TRAIN_AND_TEST.name)
    parser.add_argument('--model_name', choices=[m.name for m in SupportedModels], type=str, help='Neural network (model) to use', default=SupportedModels.VGG.name)
    parser.add_argument('--save_model', help='Save model during training', default=True)
    parser.add_argument('--save_plot_train', help='Plot VGG info during training', default=True)
    parser.add_argument('--show_plot_train', help='Plot VGG info during training', default=True)
    parser.add_argument('--plot_results', help='Plot test results', default=True)

    args = parser.parse_args()
    config = dict()
    for arg in vars(args):
        config[arg] = getattr(args, arg)
    
    if config['type'] == 'TRAIN' or config['type'] == 'TRAIN_AND_TEST':
        if config['model_name'] == 'VGG':
            train(config)
        elif config['model_name'] == 'XGBoost':
            train(config)
        else:
            raise ValueError('Please choose a valid model name: VGG or XGBoost')
            
    if config['type'] == 'TEST' or config['type'] == 'TRAIN_AND_TEST':
        if config['model_name'] == 'VGG':
            test(config)
        elif config['model_name'] == 'XGBoost':
            test(config) 
        else:
            raise ValueError('Please choose a valid model name: VGG or XGBoost')
