import enum
import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TRAIN_DATASET_PATH = 'data/dataset/ISBI2016_ISIC_Part3/ISBI2016_ISIC_Part3_Training_Data'
TEST_DATASET_PATH = 'data/dataset/ISBI2016_ISIC_Part3/ISBI2016_ISIC_Part3_Test_Data'
TRAIN_CSV_NAME = 'data/dataset/ISBI2016_ISIC_Part3/ISBI2016_ISIC_Part3_Training_GroundTruth.csv'
TEST_CSV_NAME = 'data/dataset/ISBI2016_ISIC_Part3/ISBI2016_ISIC_Part3_Test_GroundTruth.csv'

SAVED_MODEL_PATH = 'models/saved_models/'
SAVED_RESULTS_PATH = 'data/results/'

# Model constants
NUM_WORKERS = 0
BATCH_SIZE = 64
EPOCHS = 13
LEARNING_RATE = 1e-4
LR_STEP_SIZE = 5
WEIGHT_DECAY = 0

# DATASET SPECIFIC
NUM_CLASSES = 2
K_FOLD = 1

class ModelType(enum.Enum):
    TRAIN_AND_TEST = 0
    TRAIN = 1
    TEST = 2

class SupportedModels(enum.Enum):
    CNN = 0
    VGG = 1