import tensorflow as tf
from attention import CausalSelfAttention, CrossAttention
from feedforward import FeedForward
from embedding import PositionalEmbedding

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, num_heads, embedding_size, ff_dim, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()

        self.causal_self_attention = CausalSelfAttention(num_heads=num_heads, key_dim=embedding_size,
                                                         dropout=dropout_rate)
        self.cross_attention = CrossAttention(num_heads=num_heads, key_dim=embedding_size,
                                              dropout=dropout_rate)
        self.ffn = FeedForward(embedding_size, ff_dim)

    def call(self, x, context, attention_mask=None):
        x = self.causal_self_attention(x=x, attention_mask=attention_mask)
        x = self.cross_attention(x=x, context=context)
        x = self.ffn(x)
        return x


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, num_heads, ff_dim, vocab_size, embedding_size, window_size,
                 embedding_initializer, embedding_trainability=False, dropout_rate=0.1):
        super(Decoder, self).__init__()

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ff_dim = ff_dim

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.window_size = window_size

        self.embedding_initializer = embedding_initializer
        self.embedding_trainability = embedding_trainability

        self.pos_embedding = PositionalEmbedding(self.vocab_size, self.embedding_size, self.window_size,
                                                 self.embedding_initializer, self.embedding_trainability)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

        self.dec_layers = [DecoderLayer(num_heads, embedding_size, ff_dim, dropout_rate=dropout_rate)
                           for i in range(num_layers)]

    def call(self, x, context):
        # `x` is token-IDs shape (batch, target_seq_len)
        mask = self.pos_embedding.compute_mask(x)
        mask = mask[:, tf.newaxis, :]
        x = self.pos_embedding(x)  # (batch_size, target_seq_len, d_model)
        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.dec_layers[i](x, context, attention_mask=mask)

        # The shape of x is (batch_size, target_seq_len, d_model).
        return x