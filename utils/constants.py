import enum
import torch

DEVICE_NAME = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = torch.device(DEVICE_NAME)

# ISIC 2016 Task 3 Dataset Constants
TRAIN_DATASET_PATH = 'data/dataset/ISBI2016_ISIC_Part3_Training_Data'
TEST_DATASET_PATH = 'data/dataset/ISBI2016_ISIC_Part3_Test_Data'
TRAIN_CSV_NAME = 'data/dataset/ISBI2016_ISIC_Part3_Training_GroundTruth.csv'
TEST_CSV_NAME = 'data/dataset/ISBI2016_ISIC_Part3_Test_GroundTruth.csv'
VALIDATION_CSV_NAME = 'data/dataset/validation.csv'

# HAM10000 Dataset Constants
HAM10000_DATASET_PATH = 'data/dataset/archive/'
HAM10000_CSV_NAME = 'data/dataset/archive/HAM10000_metadata.csv'
HAM10000_IMAGES_1_PATH = 'HAM10000_images_part_1/'
HAM10000_IMAGES_2_PATH = 'HAM10000_images_part_2/'

SAVED_MODEL_PATH = 'models/'
SAVED_PLOT_PATH = 'results/'

# Dataset processing
MEAN_PARAMS = [0.485, 0.456, 0.406]
STD_PARAMS = [0.229, 0.224, 0.225]
RESIZE_PARAMS = (128, 128)

# Model constants
NUM_WORKERS = 0
BATCH_SIZE = 256
EPOCHS = 12
LEARNING_RATE = 1e-4
LR_STEP_SIZE = 10
WEIGHT_DECAY = 0

# DATASET SPECIFIC
NUM_CLASSES = 2
CLASS_NAMES = ['Benign', 'Malignant']
CLASS_NAMES_SERBIAN = ['Benigni tumor', 'Maligni tumor']

class ModelType(enum.Enum):
    TRAIN_AND_TEST = 0
    TRAIN = 1
    TEST = 2

class SupportedModels(enum.Enum):
    CNN = 0
    VGG = 1
    XGBoost = 2

class SupportedDataset(enum.Enum):
    ISIC = 0
    HAM = 1