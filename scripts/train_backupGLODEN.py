# ===================================================================
# PRISM-DDI: Main Training Script (Version 3 with Resume Training)
#
# This version adds the capability to automatically find and load
# the latest best model checkpoint for a given run, enabling
# seamless continuation of interrupted training.
# ===================================================================
import os
import sys
import tensorflow as tf
from datetime import datetime
import re  # Import the regular expression module

# --- 1. Project Setup and Configuration ---
# Add the project root to the Python path to allow for absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# Import our custom modules
import configs.default_config as config
from prism.dataloader import PrismDdiDataloader
from prism.model import PRISM_DDI

# Ensure all necessary result directories exist
config.ensure_directories_exist()
print("--- PRISM-DDI Training Script (with Resume Training Capability) ---")
print(f"TensorFlow Version: {tf.__version__}")
print(f"Project Root: {project_root}")


# ==============================================================================
# [NEW PART 1: Function to find the latest checkpoint and its epoch]
# ==============================================================================
def find_latest_checkpoint_and_epoch(checkpoint_dir):
    """
    Scans the checkpoint directory for run files and returns the path to the
    latest one and the epoch number to resume from.
    """
    print(f"Scanning for checkpoints in: {checkpoint_dir}")
    run_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.h5') and f.startswith('PRISM_DDI_scheduler_run')]

    if not run_files:
        print("No previous checkpoints found. Starting a new training run.")
        return None, 0
    # Find the latest run based on timestamp in the filename
    latest_run_file = sorted(run_files, reverse=True)[0]
    latest_checkpoint_path = os.path.join(checkpoint_dir, latest_run_file)
    print(f"Found latest checkpoint: {latest_run_file}")

    # --- Determine starting epoch from training history or logs (a more robust way) ---
    # For simplicity here, we assume we want to continue the last session.
    # In a real-world scenario, you might read a log file to get the exact last epoch.
    # We will start from epoch 21 as per your log. We can make this more automatic later if needed.
    # A simple way for now is to just define it. We know it was Epoch 21.
    start_epoch = 21
    print(f"Determined to resume training from epoch {start_epoch + 1}.")

    return latest_checkpoint_path, start_epoch


# --- 2. Data Loading ---
dataloader = PrismDdiDataloader(config)
train_dataset = dataloader.get_dataset(mode='train')
valid_dataset = dataloader.get_dataset(mode='valid')
print("\n--- Data Loading Complete ---")
# --- 3. Model Initialization ---
model = PRISM_DDI(config)
model.summary()
# --- 4. Optimizer and Loss Function ---
optimizer = tf.keras.optimizers.Adam(
    learning_rate=config.LEARNING_RATE,  # The scheduler will adjust this
    beta_1=config.ADAM_BETA_1,
    beta_2=config.ADAM_BETA_2,
    epsilon=config.ADAM_EPSILON
)
loss_function = tf.keras.losses.SparseCategoricalCrossentropy()
metrics = [
    tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')
]
model.compile(
    optimizer=optimizer,
    loss=loss_function,
    metrics=metrics
)
print("\n--- Model Compiled Successfully ---")
# ==============================================================================
# [NEW PART 2: Load weights and determine starting epoch]
# ==============================================================================
# Find the latest checkpoint and the epoch to start from.
# The function will determine the file name automatically.
resume_checkpoint_path, start_epoch = find_latest_checkpoint_and_epoch(config.MODEL_SAVE_PATH)
# If a checkpoint was found, load the weights into the model.
if resume_checkpoint_path:
    try:
        model.load_weights(resume_checkpoint_path)
        print(f"Successfully loaded weights from {resume_checkpoint_path}")
    except Exception as e:
        print(f"Error loading weights: {e}. Starting from scratch.")
        start_epoch = 0  # If loading fails, reset to start from epoch 0
# Use the same unique run name as the checkpoint we are resuming from.
# If starting fresh, generate a new one.
if resume_checkpoint_path:
    run_name = os.path.basename(resume_checkpoint_path).replace("_best.h5", "")
else:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"PRISM_DDI_scheduler_run_{timestamp}"
print(f"Using run name: {run_name}")
# --- 5. Callbacks Setup ---
# All callbacks will now use the consistent `run_name`.
checkpoint_path = os.path.join(config.MODEL_SAVE_PATH, f"{run_name}_best.h5")
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    verbose=1
)
early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=config.EARLY_STOPPING_PATIENCE,
    verbose=1,
    mode='max',
    restore_best_weights=True
)
lr_scheduler_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.2,
    patience=3,
    min_lr=1e-6,
    verbose=1,
    mode='max'
)
tensorboard_log_dir = os.path.join(config.LOG_PATH, run_name)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_log_dir)
callbacks = [
    model_checkpoint_callback,
    early_stopping_callback,
    lr_scheduler_callback,
    tensorboard_callback
]
print(f"\nCallbacks configured. Best model will be saved to: {checkpoint_path}")
# --- 6. Start Training ---
print("\n--- Starting/Resuming Model Training ---")
history = model.fit(
    train_dataset,
    epochs=config.EPOCHS,
    validation_data=valid_dataset,
    callbacks=callbacks,
    initial_epoch=start_epoch  # Tell Keras which epoch we are starting from!
)
print("\n--- Model Training Finished ---")
# --- 7. Final Evaluation on Test Set ---
print("\n--- Evaluating on Test Set with Best Weights ---")
test_dataloader = PrismDdiDataloader(config)
test_dataset = test_dataloader.get_dataset(mode='test')
test_results = model.evaluate(test_dataset, return_dict=True)
print(f"\nTest Set Performance:")
for metric, value in test_results.items():
    print(f"  - Test {metric}: {value:.4f}")
print("\n--- Script Finished ---")