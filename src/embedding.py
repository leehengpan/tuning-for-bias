import tensorflow as tf


def positional_encoding(length, depth):
    depth = depth / 2

    positions = tf.cast(tf.range(length)[:, tf.newaxis], dtype=tf.float32)  # (seq, 1)
    depths = tf.range(depth)[tf.newaxis, :] / depth  # (1, depth)

    angle_rates = 1 / (10000 ** depths)  # (1, depth)
    angle_rads = positions * angle_rates  # (pos, depth)

    pos_encoding = tf.concat([tf.math.sin(angle_rads), tf.math.cos(angle_rads)],
                             axis=-1)

    return tf.cast(pos_encoding, dtype=tf.float32)


class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_size, window_size, initializer, trainable=False):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.window_size = window_size
        self.initializer = initializer
        self.trainable = trainable

        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.embedding_size, mask_zero=True,
                                                   embeddings_initializer=self.initializer,
                                                   trainable=self.trainable)
        self.positional_encoding = positional_encoding(window_size, embedding_size)

    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)

    def call(self, x, add_positional_embedding=True):
        length = tf.shape(x)[1]
        if add_positional_embedding:
            return self.embedding(x) + positional_encoding(length, self.embedding_size)
        else:
            return self.embedding(x)
