# ===================================================================
# PRISM-DDI: The Main Model Definition
#
# This file defines the PRISM_DDI model, which integrates all custom
# components into a parallel dual-stream architecture.
# FINAL CORRECTED & IMPROVED VERSION
# Key changes:
# 1. ADDED LayerNormalization to stabilize inputs.
# 2. FIXED the dimension mismatch bug in SpotlightDecisionModule.
# 3. IMPROVED the SpotlightDecisionModule logic for better performance.
# ===================================================================
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Layer,
    Input,
    Dense,
    Dropout,
    LayerNormalization,
    Concatenate,
    GlobalAveragePooling1D,
    Attention,
    Embedding
)
from .layers import PrismGATLayer # Use relative import within a package
def feed_forward_network(d_model, dff):
    """A standard Feed-Forward Network block used in Transformer architectures."""
    return tf.keras.Sequential([
        Dense(dff, activation='relu'),
        Dense(d_model)
    ], name='FFN')
class PrismEncoder(Layer):
    """A single block of the PRISM-DDI encoder."""
    def __init__(self, d_model, num_heads, dff, dropout_rate, **kwargs):
        super(PrismEncoder, self).__init__(**kwargs)
        self.gat_layer = PrismGATLayer(d_model, num_heads, dropout_rate)
        self.ffn = feed_forward_network(d_model, dff)
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
    def call(self, inputs, training=None):
        x1, adj1, mask1, x2, mask2 = inputs
        
        # The GAT layer computes the update vector
        attn_output1 = self.gat_layer([x1, adj1, mask1, x2, mask2], training=training)
        attn_output1 = self.dropout1(attn_output1, training=training)
        # First residual connection (Add & Norm)
        out1 = self.layernorm1(x1 + attn_output1)
        
        # Feed-Forward network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        # Second residual connection (Add & Norm)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2
class SpotlightDecisionModule(Layer):
    """Fuses information from both streams for the final decision."""
    def __init__(self, d_model, **kwargs):
        super(SpotlightDecisionModule, self).__init__(**kwargs)
        self.d_model = d_model
        
        # Use Keras's standard Attention layer for the spotlight mechanism
        self.spotlight_attention = Attention(use_scale=True, name="spotlight_attention")
        
        # --- [BUG FIX-2: CRITICAL] ADD a projection layer for the query ---
        # This ensures the query and key dimensions match.
        self.query_projection = Dense(d_model, name="query_projection")
        # Pooling layers for creating graph-level representations
        self.pool_atomic = GlobalAveragePooling1D()
        # Final MLP for prediction
        self.dense1 = Dense(d_model, activation='relu', name="decision_dense1")
        self.dropout = Dropout(0.2, name="decision_dropout")
        
    def call(self, inputs, training=None):
        h_atomic_a, mask_atomic_a, h_atomic_b, mask_atomic_b, \
        h_motif_a, mask_motif_a, h_motif_b, mask_motif_b = inputs
        
        # --- Create graph-level summary vectors ---
        # For atomic graphs, average all node features
        v_atomic_a = self.pool_atomic(h_atomic_a, mask=mask_atomic_a)
        v_atomic_b = self.pool_atomic(h_atomic_b, mask=mask_atomic_b)
        
        # For motif graphs, use the special <global> token's representation (at index 0)
        v_motif_a = h_motif_a[:, 0, :]
        v_motif_b = h_motif_b[:, 0, :]
        
        # --- [IMPROVEMENT-2] Create a powerful global context for the query ---
        # Concatenate all four summary vectors to form a comprehensive interaction signature
        global_context = Concatenate()([v_atomic_a, v_atomic_b, v_motif_a, v_motif_b])
        # Shape: (batch, 4 * d_model)
        
        # --- [BUG FIX-2: CRITICAL] Project the context back to d_model ---
        projected_query = self.query_projection(global_context)
        # Shape: (batch, d_model)
        
        # Reshape query for the Attention layer, which expects a sequence
        query = tf.expand_dims(projected_query, 1) # Shape: (batch, 1, d_model)
        
        # --- Spotlight Attention: Use the query to find the most relevant motifs ---
        # The "values" and "keys" are the node-level motif representations from both drugs
        all_motifs_as_values = Concatenate(axis=1)([h_motif_a, h_motif_b]) # Shape: (batch, N+M, d_model)
        all_motifs_mask = Concatenate(axis=1)([mask_motif_a, mask_motif_b]) 
        
        # The attention layer will find which motifs in `all_motifs_as_values` are most
        # similar to the `query` vector, and return a weighted average.
        context_vector = self.spotlight_attention(
            [query, all_motifs_as_values], 
            mask=[None, all_motifs_mask] # Mask is applied to the values/keys
        )
        
        # Reshape context vector back to a flat tensor for the MLP
        final_rep = tf.squeeze(context_vector, axis=1) # Shape: (batch, d_model)
        
        # Final decision-making MLP
        x = self.dense1(final_rep)
        x = self.dropout(x, training=training)
        
        return x
def PRISM_DDI(config):
    """The factory function to build the complete PRISM-DDI Keras Model."""
    
    # --- Input Layers ---
    input_atomic_features_a = Input(shape=(None, config.ATOM_FEATURE_DIM), name="atomic_features_a")
    input_atomic_adj_a = Input(shape=(None, None), name="atomic_adj_a")
    input_atomic_features_b = Input(shape=(None, config.ATOM_FEATURE_DIM), name="atomic_features_b")
    input_atomic_adj_b = Input(shape=(None, None), name="atomic_adj_b")
    
    input_motif_ids_a = Input(shape=(None,), dtype=tf.int32, name="motif_ids_a")
    input_motif_adj_a = Input(shape=(None, None), name="motif_adj_a")
    input_motif_ids_b = Input(shape=(None,), dtype=tf.int32, name="motif_ids_b")
    input_motif_adj_b = Input(shape=(None, None), name="motif_adj_b")
    
    # --- [IMPROVEMENT-1] Initial Embeddings and Normalization ---
    # Atomic Stream
    atomic_embedding_layer = Dense(config.D_MODEL, activation='relu', name="atomic_feature_embedding")
    norm_atomic = LayerNormalization(name="norm_atomic_input")
    x_atomic_a_init = norm_atomic(atomic_embedding_layer(input_atomic_features_a))
    x_atomic_b_init = norm_atomic(atomic_embedding_layer(input_atomic_features_b))
    
    # Motif Stream
    motif_embedding_layer = Embedding(config.MOTIF_VOCAB_SIZE, config.D_MODEL, name="motif_embedding")
    norm_motif = LayerNormalization(name="norm_motif_input")
    x_motif_a_init = norm_motif(motif_embedding_layer(input_motif_ids_a))
    x_motif_b_init = norm_motif(motif_embedding_layer(input_motif_ids_b))
    
    # --- Create Padding Masks (boolean type is correct for TF > 2.4) ---
    mask_atomic_a = tf.reduce_sum(tf.abs(input_atomic_features_a), axis=-1) > 0
    mask_atomic_b = tf.reduce_sum(tf.abs(input_atomic_features_b), axis=-1) > 0
    mask_motif_a = input_motif_ids_a > 0
    mask_motif_b = input_motif_ids_b > 0
    
    # --- Parallel Backbone Encoders ---
    # Create shared encoder layers for both streams
    atomic_encoders = [PrismEncoder(config.D_MODEL, config.NUM_HEADS, config.D_FF, config.DROPOUT_RATE, name=f"atomic_encoder_{i}") for i in range(config.NUM_ENCODER_LAYERS)]
    motif_encoders = [PrismEncoder(config.D_MODEL, config.NUM_HEADS, config.D_FF, config.DROPOUT_RATE, name=f"motif_encoder_{i}") for i in range(config.NUM_ENCODER_LAYERS)]
    # Process Atomic Stream
    x_atomic_a, x_atomic_b = x_atomic_a_init, x_atomic_b_init
    for encoder in atomic_encoders:
        # Symmetrically update both drug representations
        temp_a = encoder([x_atomic_a, input_atomic_adj_a, mask_atomic_a, x_atomic_b, mask_atomic_b])
        temp_b = encoder([x_atomic_b, input_atomic_adj_b, mask_atomic_b, x_atomic_a, mask_atomic_a])
        x_atomic_a, x_atomic_b = temp_a, temp_b
    # Process Motif Stream
    x_motif_a, x_motif_b = x_motif_a_init, x_motif_b_init
    for encoder in motif_encoders:
        # Symmetrically update both drug representations
        temp_a = encoder([x_motif_a, input_motif_adj_a, mask_motif_a, x_motif_b, mask_motif_b])
        temp_b = encoder([x_motif_b, input_motif_adj_b, mask_motif_b, x_motif_a, mask_motif_a])
        x_motif_a, x_motif_b = temp_a, temp_b
        
    # --- Spotlight Decision Module for Fusion and Final Prediction ---
    spotlight_module = SpotlightDecisionModule(config.D_MODEL)
    decision_vector = spotlight_module([
        x_atomic_a, mask_atomic_a, x_atomic_b, mask_atomic_b,
        x_motif_a, mask_motif_a, x_motif_b, mask_motif_b
    ])
    
    # --- Final Classifier Head ---
    predictions = Dense(config.NUM_CLASSES, activation='softmax', name="final_classifier")(decision_vector)
    
    # --- Build the Keras Model ---
    model = Model(
        inputs={
            "atomic_features_a": input_atomic_features_a, "atomic_adj_a": input_atomic_adj_a,
            "atomic_features_b": input_atomic_features_b, "atomic_adj_b": input_atomic_adj_b,
            "motif_ids_a": input_motif_ids_a, "motif_adj_a": input_motif_adj_a,
            "motif_ids_b": input_motif_ids_b, "motif_adj_b": input_motif_adj_b
        },
        outputs=predictions
    )
    
    return model