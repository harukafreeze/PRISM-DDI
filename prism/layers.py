# ===================================================================
# PRISM-DDI: Custom Keras Layers
#
# Contains the core PrismGATLayer.
# FINAL CORRECTED & IMPROVED VERSION
# Key changes:
# 1. FIXED the attention mask logic in scaled_dot_product_attention.
# 2. IMPROVED the Gating mechanism to include original state `x_self`.
# ===================================================================
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Dropout
from tensorflow.keras import activations
class PrismGATLayer(Layer):
    """
    The core computational layer of the PRISM-DDI model.
    Processes a "self" drug's graph by attending to its local neighborhood
    and to its interacting partner drug.
    """
    def __init__(self, d_model, num_heads, dropout_rate=0.1, **kwargs):
        super(PrismGATLayer, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        
        self.depth = d_model // self.num_heads
        
        # Projections for self-attention/GAT channel
        self.wq_self = Dense(d_model, name="wq_self")
        self.wk_self = Dense(d_model, name="wk_self")
        self.wv_self = Dense(d_model, name="wv_self")
        self.dense_self = Dense(d_model, name="dense_self")
        
        # Projections for cross-attention channel
        self.wq_cross = Dense(d_model, name="wq_cross")
        self.wk_cross = Dense(d_model, name="wk_cross")
        self.wv_cross = Dense(d_model, name="wv_cross")
        self.dense_cross = Dense(d_model, name="dense_cross")
        
        # Gating mechanism
        # The gate's input will be a concatenation of self-update, cross-update, and original state.
        self.gate_dense = Dense(d_model, activation='sigmoid', name="gate_dense")
        
        self.dropout_self = Dropout(dropout_rate)
        self.dropout_cross = Dropout(dropout_rate)
    def split_heads(self, x, batch_size):
        """Reshapes the input for multi-head attention."""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    def scaled_dot_product_attention(self, q, k, v, mask, adj=None):
        """
        Calculates attention scores, applying adjacency and padding masks.
        """
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        
        # Scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        # Add adjacency mask if provided
        if adj is not None:
            # Add a large negative number to positions where there is NO edge (adj=0)
            adj_mask = (1.0 - tf.cast(adj, tf.float32)) * -1e9
            scaled_attention_logits += tf.expand_dims(adj_mask, axis=1) # Add head dimension for broadcasting
        # --- [BUG FIX-1: CRITICAL] CORRECT ATTENTION MASK LOGIC ---
        if mask is not None:
            # The input mask is boolean (True for real tokens, False for padding).
            # 1. Cast it to float32. True -> 1.0, False -> 0.0
            mask = tf.cast(mask, tf.float32)
            # 2. Reshape to be broadcastable with attention scores.
            mask = mask[:, tf.newaxis, tf.newaxis, :] # Shape: (batch, 1, 1, seq_len)
            # 3. Add a large negative number to PADDING positions (where mask is 0.0),
            #    by using (1.0 - mask).
            scaled_attention_logits += (1.0 - mask) * -1e9
            
        attention_weights = activations.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)
        return output
    def call(self, inputs, training=None):
        # Unpack inputs
        x_self, adj_self, mask_self, x_cross, mask_cross = inputs
        batch_size = tf.shape(x_self)[0]
        # --- Channel 1: Intra-Drug Attention (Self-Attention over Local Graph) ---
        q_s, k_s, v_s = self.wq_self(x_self), self.wk_self(x_self), self.wv_self(x_self)
        q_s = self.split_heads(q_s, batch_size)
        k_s = self.split_heads(k_s, batch_size)
        v_s = self.split_heads(v_s, batch_size)
        
        attention_output_self = self.scaled_dot_product_attention(q_s, k_s, v_s, mask_self, adj_self)
        attention_output_self = tf.transpose(attention_output_self, perm=[0, 2, 1, 3])
        concat_attention_self = tf.reshape(attention_output_self, (batch_size, -1, self.d_model))
        output_self = self.dense_self(concat_attention_self)
        output_self = self.dropout_self(output_self, training=training)
        # --- Channel 2: Inter-Drug Attention (Cross-Attention to Partner Drug) ---
        q_c, k_c, v_c = self.wq_cross(x_self), self.wk_cross(x_cross), self.wv_cross(x_cross)
        q_c = self.split_heads(q_c, batch_size)
        k_c = self.split_heads(k_c, batch_size)
        v_c = self.split_heads(v_c, batch_size)
        attention_output_cross = self.scaled_dot_product_attention(q_c, k_c, v_c, mask_cross) # No adjacency matrix for cross attention
        attention_output_cross = tf.transpose(attention_output_cross, perm=[0, 2, 1, 3])
        concat_attention_cross = tf.reshape(attention_output_cross, (batch_size, -1, self.d_model))
        output_cross = self.dense_cross(concat_attention_cross)
        output_cross = self.dropout_cross(output_cross, training=training)
        
        # --- [IMPROVEMENT-3] Dynamic Gating Fusion (Optimized) ---
        # The gate's decision is now based on the original state and the two proposed updates.
        gate_input = tf.concat([x_self, output_self, output_cross], axis=-1)
        gate = self.gate_dense(gate_input)
        
        # The final output is a gated combination of the two new representations
        fused_output = (1.0 - gate) * output_self + gate * output_cross
        
        return fused_output