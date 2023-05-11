import tensorflow as tf
from encoder import Encoder
from decoder import Decoder


class TransformerModel(tf.keras.Model):
    def __init__(self, num_layers, num_heads, ff_dim, embedding_size,
                 content_vocab_size, title_vocab_size, content_window_size, title_window_size,
                 content_embedding_initializer, title_embedding_initializer,
                 content_embedding_trainability, title_embedding_trainability,
                 dropout_rate=0.1):
        super().__init__()
        self.encoder = Encoder(num_layers, num_heads, ff_dim, content_vocab_size, embedding_size,
                               content_window_size, content_embedding_initializer, content_embedding_trainability,
                               dropout_rate)
        self.decoder = Decoder(num_layers, num_heads, ff_dim, title_vocab_size, embedding_size,
                               title_window_size, title_embedding_initializer, title_embedding_trainability,
                               dropout_rate)

        self.dense_layer = tf.keras.layers.Dense(title_vocab_size)

    def call(self, inputs):
        content, title = inputs
        print('At transformer call: ', content==None, title==None)
        context = self.encoder(content)
        output = self.decoder(title, context)
        logits = self.dense_layer(output)
        return logits
