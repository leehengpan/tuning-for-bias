import tensorflow as tf
from embedding import PositionalEmbedding
from attention import GlobalSelfAttention
from feedforward import FeedForward


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, num_heads, embedding_size, ff_dim, dropout_rate=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.embedding_size = embedding_size
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate

        self.self_attention = GlobalSelfAttention(num_heads=num_heads, key_dim=embedding_size,
                                                  dropout=dropout_rate)
        self.ffn = FeedForward(embedding_size, ff_dim)

    def call(self, x, attention_mask=None):
        x = self.self_attention(x, attention_mask=attention_mask)
        x = self.ffn(x)
        return x


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, num_heads, ff_dim, vocab_size, embedding_size,
                 window_size, embedding_initializer, embedding_trainability=False, dropout_rate=0.1):
        super().__init__()

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

        self.enc_layers = [EncoderLayer(num_heads, embedding_size, ff_dim, dropout_rate) for i in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x):
        # x is tokenized numerical values
        mask = self.pos_embedding.compute_mask(x)
        mask = mask[:, tf.newaxis, :]
        x = self.pos_embedding(x)

        # Add dropout.
        x = self.dropout(x)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, attention_mask=mask)
        return x
