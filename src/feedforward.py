import tensorflow as tf

class FeedForward(tf.keras.layers.Layer):
    def __init__(self, embedding_size, ff_dim, dropout_rate=0.1):
        super().__init__()
        self.seq = tf.keras.Sequential([tf.keras.layers.Dense(ff_dim, activation='relu'),
                                        tf.keras.layers.Dense(embedding_size),
                                        tf.keras.layers.Dropout(dropout_rate)])
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x)
        return x