# scripts/train.py (Slightly refactored to be importable)
import os
import sys
import tensorflow as tf
from datetime import datetime
# --- 1. Project Setup and Configuration ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
import configs.default_config as config
from prism.dataloader import PrismDdiDataloader
from prism.model import PRISM_DDI
# ===================================================================
# [REFACTOR] All logic is now inside a main() function.
# ===================================================================
def main(ablation_name=None):
    """
    The main training and evaluation function.
    Args:
        ablation_name (str, optional): An identifier for the ablation run.
                                       If provided, it's added to the run name.
    """
    config.ensure_directories_exist()
    print("--- PRISM-DDI Training Script ---")
    if ablation_name:
        print(f"--- Running in Ablation Mode: {ablation_name} ---")
    
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
        learning_rate=config.LEARNING_RATE,
        # ... other optimizer params
    )
    loss_function = tf.keras.losses.SparseCategoricalCrossentropy()
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')]
    
    model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)
    print("\n--- Model Compiled Successfully ---")
    # --- 5. Callbacks Setup ---
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name_base = "PRISM_DDI_scheduler_run"
    if ablation_name:
        run_name = f"{run_name_base}_ablation_{ablation_name}_{timestamp}"
    else:
        run_name = f"{run_name_base}_{timestamp}"
    # ... The rest of your callbacks setup code is GREAT, just use `run_name` ...
    checkpoint_path = os.path.join(config.MODEL_SAVE_PATH, f"{run_name}_best.h5")
    # ... (ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard setup as before)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, save_weights_only=True, monitor='val_accuracy',
        mode='max', save_best_only=True, verbose=1
    )
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=config.EARLY_STOPPING_PATIENCE,
        verbose=1, mode='max', restore_best_weights=True
    )
    lr_scheduler_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_accuracy', factor=0.2, patience=3,
        min_lr=1e-6, verbose=1, mode='max'
    )
    tensorboard_log_dir = os.path.join(config.LOG_PATH, run_name)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_log_dir)
    callbacks = [ model_checkpoint_callback, early_stopping_callback, 
                  lr_scheduler_callback, tensorboard_callback ]
    print(f"\nCallbacks configured. Best model will be saved to: {checkpoint_path}")
    # --- 6. Start Training ---
    print("\n--- Starting Model Training ---")
    model.fit(
        train_dataset,
        epochs=config.EPOCHS,
        validation_data=valid_dataset,
        callbacks=callbacks
    )
    print("\n--- Model Training Finished ---")
    # --- 7. Final Evaluation on Test Set ---
    print("\n--- Evaluating on Test Set with Best Weights ---")
    test_dataloader = PrismDdiDataloader(config)
    test_dataset = test_dataloader.get_dataset(mode='test')
    test_results = model.evaluate(test_dataset, return_dict=True)
    
    print(f"\nFinal Test Set Performance for run '{run_name}':")
    for metric, value in test_results.items():
        print(f"  - Test {metric}: {value:.4f}")
    print("\n--- Script Finished ---")
# ===================================================================
# This block allows the script to be run directly as before.
# ===================================================================
if __name__ == '__main__':
    main()