import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Chemins
TRAIN_DIR_A = "data/trainA"
TRAIN_DIR_B = "data/trainB"
VAL_DIR_A   = "data/testA"
VAL_DIR_B   = "data/testB"
OUTPUT_DIR  = "outputs"
CHECKPOINT_DIR = "checkpoints"

# Hyperparamètres
BATCH_SIZE     = 1
IMAGE_SIZE     = 256
LEARNING_RATE  = 2e-4
LAMBDA_CYCLE   = 10
LAMBDA_IDENTITY = 0.5
NUM_EPOCHS     = 30
NUM_WORKERS    = 2

# Sauvegarde
LOAD_MODEL = True
SAVE_MODEL = True