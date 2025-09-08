# ===================================================================
# PRISM-DDI: Default Configuration
#
# This file serves as the single source of truth for all hyperparameters,
# file paths, and model settings.
# THIS VERSION IS CONFIGURED FOR OFFICIAL, FULL-SCALE TRAINING.
# ===================================================================
import os
# --- 1. File and Directory Paths ---
# We use os.path.join for cross-platform compatibility.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Raw Data
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw')
TRAIN_FILE = os.path.join(RAW_DATA_PATH, 'tr_dataset.csv')
VALID_FILE = os.path.join(RAW_DATA_PATH, 'val_dataset.csv')
TEST_FILE = os.path.join(RAW_DATA_PATH, 'tst_dataset.csv')
TOKEN_ID_FILE = os.path.join(RAW_DATA_PATH, 'token_id.json')
# Processed Data
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed')
PRECOMPUTED_FILE_PATH = os.path.join(PROCESSED_DATA_PATH, 'precomputed_drug_features.npy')
# Results
RESULTS_PATH = os.path.join(PROJECT_ROOT, 'results')
MODEL_SAVE_PATH = os.path.join(RESULTS_PATH, 'trained_models')
LOG_PATH = os.path.join(RESULTS_PATH, 'logs')
FIGURE_PATH = os.path.join(RESULTS_PATH, 'figures')
# --- 2. Data Preprocessing Parameters ---
# The feature dimension for a single atom, determined from the preprocessing step.
ATOM_FEATURE_DIM = 77  
# --- 3. Model Hyperparameters (Official Settings) ---
# Global settings
NUM_CLASSES = 4        # e.g., for ZhongDDI (0: inhibit, 1: induce, 2: inhibit_rev, 3: induce_rev)
MOTIF_VOCAB_SIZE = 287 # Your vocabulary size (283 motifs + 4 special tokens)
# RGDA-IP Encoder Backbone Settings
NUM_ENCODER_LAYERS = 6    # Number of stacked encoder layers. A solid starting point.
D_MODEL = 256            # The core dimensionality of the model's hidden states.
NUM_HEADS = 8              # Number of attention heads.
D_FF = D_MODEL * 4         # Dimensionality of the inner layer of the Feed-Forward Network (usually 4*d_model).
DROPOUT_RATE = 0.1       # Dropout rate for regularization.
# --- 4. Training Hyperparameters (Official Settings) ---
# Training loop
EPOCHS = 100               # Maximum number of epochs. Early stopping will likely stop it sooner.
BATCH_SIZE = 64            # A standard batch size for large models. Adjust if you face memory issues.
# Optimizer (Adam)
LEARNING_RATE = 1e-4       # A robust learning rate for Transformer-based models.
ADAM_BETA_1 = 0.9
ADAM_BETA_2 = 0.98
ADAM_EPSILON = 1e-9
# Callbacks
EARLY_STOPPING_PATIENCE = 10 # Stop training if validation accuracy doesn't improve for 10 epochs.
# --- 5. Utility function to create necessary directories ---
def ensure_directories_exist():
    """Creates all necessary result directories if they don't exist."""
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(LOG_PATH, exist_ok=True)
    os.makedirs(FIGURE_PATH, exist_ok=True)
    print("Project directories are ensured to exist.")