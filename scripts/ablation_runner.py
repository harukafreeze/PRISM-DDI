# scripts/ablation_runner.py
#
# A dedicated, robust script for conducting ablation studies on the PRISM-DDI model.
# It uses the monkey patching technique to dynamically alter model behavior at runtime.
#
# USAGE EXAMPLES:
# (Run from the project root directory: PRISM-DDI/)
#
# 1. Run Motif-Only experiment:
#    python scripts/ablation_runner.py --ablation_type motif_only
#
# 2. Run Atomic-Only experiment:
#    python scripts/ablation_runner.py --ablation_type atomic_only
#
# 3. Run No-Cross-Attention experiment:
#    python scripts/ablation_runner.py --ablation_type no_cross_attention
#
# 4. Run No-Spotlight experiment:
#    python scripts/ablation_runner.py --ablation_type no_spotlight
#
# 5. Run No-Gating (Addition Fusion) experiment:
#    python scripts/ablation_runner.py --ablation_type no_gating
#
import os
import sys
import argparse
import tensorflow as tf

# --- 1. Project Setup to allow imports ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# Import necessary components from the project
from prism.layers import PrismGATLayer
from prism.model import SpotlightDecisionModule

# --- We will import the main training function, but we need to do it carefully ---
# In Python, imports are cached. To make monkey patching work, we need to apply
# the patch BEFORE the module that uses it is imported for the first time.
# So, we will import the main training script's function just before we need it.
print("--- PRISM-DDI Ablation Runner Initialized ---")
# --- 2. Store original methods for restoration ---
original_gat_call = PrismGATLayer.call
original_spotlight_call = SpotlightDecisionModule.call
print("Original model methods have been saved.")


# --- 3. Define the Ablation Logic (Modified `call` methods) ---
# Each function here represents a "surgical" modification to a layer.
def setup_ablation_motif_only():
    """
    Modifies SpotlightDecisionModule to ignore the atomic stream.
    The model effectively becomes a single-stream (motif-only) model.
    """
    print("\n[ABLATION PATCH] Activating 'Motif-Only' mode.")

    def patched_spotlight_call(self, inputs, training=None):
        # Unpack all inputs, but we will ignore the atomic ones.
        h_atomic_a, mask_atomic_a, h_atomic_b, mask_atomic_b, \
            h_motif_a, mask_motif_a, h_motif_b, mask_motif_b = inputs

        # Use only motif representations for the 'global context' query
        v_motif_a = h_motif_a[:, 0, :]
        v_motif_b = h_motif_b[:, 0, :]
        global_context = tf.keras.layers.Concatenate()([v_motif_a, v_motif_b])

        # --- The rest of the logic remains similar but simplified ---
        # The key difference is the query does not contain atomic info.
        projected_query = self.query_projection(global_context)  # Project from 2*D_MODEL
        query = tf.expand_dims(projected_query, 1)
        all_motifs = tf.keras.layers.Concatenate(axis=1)([h_motif_a, h_motif_b])
        all_motifs_mask = tf.keras.layers.Concatenate(axis=1)([mask_motif_a, mask_motif_b])
        context_vector = self.spotlight_attention([query, all_motifs], mask=[None, all_motifs_mask])
        final_rep = tf.squeeze(context_vector, axis=1)
        x = self.dense1(final_rep)
        x = self.dropout(x, training=training)
        return x

    SpotlightDecisionModule.call = patched_spotlight_call


def setup_ablation_atomic_only():
    """
    Modifies SpotlightDecisionModule to ignore the motif stream.
    The model becomes single-stream (atomic-only) and loses the Spotlight mechanism.
    """
    print("\n[ABLATION PATCH] Activating 'Atomic-Only' mode.")

    def patched_spotlight_call(self, inputs, training=None):
        # We only use the atomic inputs
        h_atomic_a, mask_atomic_a, h_atomic_b, mask_atomic_b, _, _, _, _ = inputs

        # Create graph-level representations from the atomic stream
        v_atomic_a = self.pool_atomic(h_atomic_a, mask=mask_atomic_a)
        v_atomic_b = self.pool_atomic(h_atomic_b, mask=mask_atomic_b)
        # Since there are no motifs to attend to, we bypass the Spotlight attention.
        # We just concatenate the global atomic vectors and feed them to the MLP.
        final_rep = tf.keras.layers.Concatenate()([v_atomic_a, v_atomic_b])

        x = self.dense1(final_rep)
        x = self.dropout(x, training=training)
        return x

    SpotlightDecisionModule.call = patched_spotlight_call


def setup_ablation_no_cross_attention():
    """
    Modifies PrismGATLayer to disable the cross-attention channel.
    The GAT layer now only performs self-attention within each drug's graph.
    """
    print("\n[ABLATION PATCH] Activating 'No Cross-Attention' mode.")

    def patched_gat_call(self, inputs, training=None):
        x_self, adj_self, mask_self, x_cross, mask_cross = inputs
        batch_size = tf.shape(x_self)[0]
        # --- Channel 1: Intra-Drug Attention (Self-Attention) ---
        # This part remains the same as the original.
        q_s, k_s, v_s = self.wq_self(x_self), self.wk_self(x_self), self.wv_self(x_self)
        q_s, k_s, v_s = self.split_heads(q_s, batch_size), self.split_heads(k_s, batch_size), self.split_heads(v_s,
                                                                                                               batch_size)
        attention_output_self = self.scaled_dot_product_attention(q_s, k_s, v_s, mask_self, adj_self)
        attention_output_self = tf.transpose(attention_output_self, perm=[0, 2, 1, 3])
        concat_attention_self = tf.reshape(attention_output_self, (batch_size, -1, self.d_model))
        output_self = self.dense_self(concat_attention_self)
        output_self = self.dropout_self(output_self, training=training)
        # --- Ablation: Bypassing cross-attention and gating ---
        # The fused output is just the self-attention output.
        fused_output = output_self
        return fused_output

    PrismGATLayer.call = patched_gat_call


def setup_ablation_no_spotlight():
    """
    Modifies SpotlightDecisionModule to use simple pooling and concatenation
    instead of the attention-based spotlight mechanism.
    """
    print("\n[ABLATION PATCH] Activating 'No Spotlight (Simple Pooling)' mode.")

    def patched_spotlight_call(self, inputs, training=None):
        # We use all inputs, but process them in a simpler way.
        h_atomic_a, mask_atomic_a, h_atomic_b, mask_atomic_b, \
            h_motif_a, mask_motif_a, h_motif_b, mask_motif_b = inputs
        # Create graph-level summary vectors for all streams
        v_atomic_a = self.pool_atomic(h_atomic_a, mask=mask_atomic_a)
        v_atomic_b = self.pool_atomic(h_atomic_b, mask=mask_atomic_b)
        v_motif_a = h_motif_a[:, 0, :]
        v_motif_b = h_motif_b[:, 0, :]
        # Concatenate all summary vectors directly
        final_rep = tf.keras.layers.Concatenate()([v_atomic_a, v_atomic_b, v_motif_a, v_motif_b])
        # Feed into the final MLP
        x = self.dense1(final_rep)
        x = self.dropout(x, training=training)
        return x

    SpotlightDecisionModule.call = patched_spotlight_call


def setup_ablation_no_gating():
    """
    Modifies PrismGATLayer to fuse self- and cross-attention via simple addition
    instead of the learned gating mechanism.
    """
    print("\n[ABLATION PATCH] Activating 'No Gating (Addition Fusion)' mode.")

    def patched_gat_call(self, inputs, training=None):
        # --- This `call` method is identical to the original, EXCEPT for the fusion part ---
        x_self, adj_self, mask_self, x_cross, mask_cross = inputs
        batch_size = tf.shape(x_self)[0]
        # Self-Attention channel
        q_s, k_s, v_s = self.wq_self(x_self), self.wk_self(x_self), self.wv_self(x_self)
        q_s, k_s, v_s = self.split_heads(q_s, batch_size), self.split_heads(k_s, batch_size), self.split_heads(v_s,
                                                                                                               batch_size)
        attention_output_self = self.scaled_dot_product_attention(q_s, k_s, v_s, mask_self, adj_self)
        attention_output_self = tf.transpose(attention_output_self, perm=[0, 2, 1, 3])
        concat_attention_self = tf.reshape(attention_output_self, (batch_size, -1, self.d_model))
        output_self = self.dense_self(concat_attention_self)
        output_self = self.dropout_self(output_self, training=training)
        # Cross-Attention channel
        q_c, k_c, v_c = self.wq_cross(x_self), self.wk_cross(x_cross), self.wv_cross(x_cross)
        q_c, k_c, v_c = self.split_heads(q_c, batch_size), self.split_heads(k_c, batch_size), self.split_heads(v_c,
                                                                                                               batch_size)
        attention_output_cross = self.scaled_dot_product_attention(q_c, k_c, v_c, mask_cross)
        attention_output_cross = tf.transpose(attention_output_cross, perm=[0, 2, 1, 3])
        concat_attention_cross = tf.reshape(attention_output_cross, (batch_size, -1, self.d_model))
        output_cross = self.dense_cross(concat_attention_cross)
        output_cross = self.dropout_cross(output_cross, training=training)
        # --- Ablation: Fusing with simple addition ---
        fused_output = output_self + output_cross
        return fused_output

    PrismGATLayer.call = patched_gat_call


# --- 4. Main Execution Block ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run ablation studies for the PRISM-DDI model.")
    parser.add_argument(
        "--ablation_type",
        type=str,
        required=True,
        choices=[
            'motif_only',
            'atomic_only',
            'no_cross_attention',
            'no_spotlight',
            'no_gating'
        ],
        help="Specify the ablation experiment to run."
    )
    args = parser.parse_args()
    # --- Apply the selected patch dynamically ---
    if args.ablation_type == 'motif_only':
        setup_ablation_motif_only()
    elif args.ablation_type == 'atomic_only':
        setup_ablation_atomic_only()
    elif args.ablation_type == 'no_cross_attention':
        setup_ablation_no_cross_attention()
    elif args.ablation_type == 'no_spotlight':
        setup_ablation_no_spotlight()
    elif args.ablation_type == 'no_gating':
        setup_ablation_no_gating()

    # --- Now we can safely import and run the training script ---
    # This ensures the patch is active when the model is built.
    try:
        from scripts.train import main as run_training  # Assuming your train.py has a main() function

        print(f"\n--- Starting Ablation Experiment: {args.ablation_type} ---\n")
        # To make the train.py script compatible, we'll wrap its content in a main() function.
        # This is a good practice for making scripts reusable.
        run_training(ablation_name=args.ablation_type)

    except Exception as e:
        print(f"\nAN ERROR OCCURRED DURING ABLATION TRAINING: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # --- 5. Critical Cleanup: Restore original methods ---
        PrismGATLayer.call = original_gat_call
        SpotlightDecisionModule.call = original_spotlight_call
        print(f"\n--- Ablation Experiment Finished: {args.ablation_type} ---")
        print("Original model methods have been restored.")