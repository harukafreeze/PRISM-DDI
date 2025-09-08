# ===================================================================
# PRISM-DDI Dataloader
#
# This file defines the data loading pipeline for the PRISM-DDI model.
# It reads the precomputed drug features and prepares batches for training,
# validation, and testing, handling dynamic padding.
# FINAL CORRECTED VERSION
# ===================================================================
import tensorflow as tf
import numpy as np
import pandas as pd
class PrismDdiDataloader:
    """
    Manages the data loading pipeline for the PRISM-DDI model.
    """
    def __init__(self, config):
        """
        Initializes the dataloader.
        
        Args:
            config: A configuration object containing paths and parameters.
        """
        self.config = config
        
        print("--- Initializing Dataloader ---")
        # 1. Load the massive precomputed features dictionary into memory
        print(f"Loading precomputed features from: {config.PRECOMPUTED_FILE_PATH}")
        self.precomputed_data = np.load(config.PRECOMPUTED_FILE_PATH, allow_pickle=True).item()
        print("Precomputed features loaded successfully.")
        
        # 2. Load the datasets that define the pairs and labels
        self.df_train = pd.read_csv(config.TRAIN_FILE)
        self.df_valid = pd.read_csv(config.VALID_FILE)
        self.df_test = pd.read_csv(config.TEST_FILE)
        print("Train/Valid/Test CSVs loaded.")
    def _get_drug_data(self, smiles_a, smiles_b):
        """
        Internal function to retrieve precomputed data for a drug pair.
        This runs in Python/Numpy.
        """
        smiles_a = smiles_a.numpy().decode('utf-8')
        smiles_b = smiles_b.numpy().decode('utf-8')
        
        data_a = self.precomputed_data[smiles_a]
        data_b = self.precomputed_data[smiles_b]
        
        # We return only the tensors that are directly used as model inputs in the first version
        return (
            data_a['atomic_features'], data_a['atomic_adj'],
            data_a['motif_ids'], data_a['motif_adj'],
            data_b['atomic_features'], data_b['atomic_adj'],
            data_b['motif_ids'], data_b['motif_adj']
        )
    def _map_fn_wrapper(self, smiles_a, smiles_b, label):
        """
        Wraps the Python data retrieval function in tf.py_function and
        formats the output into a dictionary.
        """
        output_types = [
            tf.float32, tf.float32, tf.int32, tf.float32, # Drug A
            tf.float32, tf.float32, tf.int32, tf.float32  # Drug B
        ]
        
        inputs_tuple = tf.py_function(self._get_drug_data, [smiles_a, smiles_b], output_types)
        
        # Convert the tuple of tensors into a dictionary
        inputs_dict = {
            "atomic_features_a": inputs_tuple[0],
            "atomic_adj_a": inputs_tuple[1],
            "motif_ids_a": inputs_tuple[2],
            "motif_adj_a": inputs_tuple[3],
            "atomic_features_b": inputs_tuple[4],
            "atomic_adj_b": inputs_tuple[5],
            "motif_ids_b": inputs_tuple[6],
            "motif_adj_b": inputs_tuple[7]
        }
        
        return inputs_dict, tf.cast(label, tf.int32)
    def get_dataset(self, mode='train'):
        """
        Gets the fully prepared and batched tf.data.Dataset for a specific mode.
        """
        if mode == 'train':
            df = self.df_train
            batch_size = self.config.BATCH_SIZE
            shuffle = True
        elif mode == 'valid':
            df = self.df_valid
            batch_size = self.config.BATCH_SIZE
            shuffle = False
        elif mode == 'test':
            df = self.df_test
            batch_size = self.config.BATCH_SIZE
            shuffle = False
        else:
            raise ValueError("Mode must be 'train', 'valid', or 'test'.")
            
        print(f"--- Preparing dataset for mode: {mode} ---")
            
        dataset = tf.data.Dataset.from_tensor_slices((
            df['drug_A'].values,
            df['drug_B'].values,
            # Use the correct column name for labels as per MeTDDI data. It's often 'DDI'.
            # If your CSV has a different name, change it here.
            df['DDI'].values 
        ))
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(df), reshuffle_each_iteration=True)
            
        # Apply the mapping function to load data and create the dictionary structure
        dataset = dataset.map(self._map_fn_wrapper, num_parallel_calls=tf.data.AUTOTUNE)
        
        # Padded batching with the CORRECTED dictionary and label structure
        dataset = dataset.padded_batch(
            batch_size=batch_size,
            # This is a 2-element tuple: (padded_shape_for_dict, padded_shape_for_label)
            padded_shapes=(
                { # Part 1: Shapes for the input dictionary
                  "atomic_features_a": [None, self.config.ATOM_FEATURE_DIM],
                  "atomic_adj_a": [None, None],
                  "motif_ids_a": [None],
                  "motif_adj_a": [None, None],
                  "atomic_features_b": [None, self.config.ATOM_FEATURE_DIM],
                  "atomic_adj_b": [None, None],
                  "motif_ids_b": [None],
                  "motif_adj_b": [None, None]
                },
                # Part 2: Shape for the scalar label
                tf.TensorShape([]) 
            ),
            # This is a 2-element tuple: (padding_values_for_dict, padding_value_for_label)
            padding_values=(
                 { # Part 1: Padding values for the input dictionary
                  "atomic_features_a": 0.0, "atomic_adj_a": 0.0,
                  "motif_ids_a": 0, "motif_adj_a": 0.0,
                  "atomic_features_b": 0.0, "atomic_adj_b": 0.0,
                  "motif_ids_b": 0, "motif_adj_b": 0.0
                 },
                 # Part 2: Padding value for the label
                 tf.constant(0, dtype=tf.int32)
            ),
            drop_remainder=False
        )
        
        # Improve performance by prefetching data
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        
        print(f"Dataset for mode '{mode}' prepared with batch size {batch_size}.")
        return dataset