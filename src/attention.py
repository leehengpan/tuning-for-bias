import tensorflow as tf

class BaseAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()

class CrossAttention(BaseAttention):
    def call(self, x, context):
        attn_output = self.mha(query=x, key=context, value=context)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x

class GlobalSelfAttention(BaseAttention):
    def call(self, x, attention_mask=None):
        attn_output = self.mha(query=x, value=x, key=x, attention_mask=attention_mask)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x

class CausalSelfAttention(BaseAttention):
    def call(self, x, attention_mask=None):
        attn_output = self.mha(query=x, value=x, key=x, use_causal_mask=True, attention_mask=attention_mask)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x