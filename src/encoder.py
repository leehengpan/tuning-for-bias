import tensorflow as tf

# Encoder block class for transformer model 
class EncoderBlock(tf.keras.layers.Layer):
    def __init__(self, emb_sz, num_heads, key_dim, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)

        # dense layers for encoder block --> NOTE TO GENERALIZE 
        self.ff_layer = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(emb_sz)
        ])
        
        # self attention layer 
        self.self_atten = tf.keras.layers.MultiHeadAttention(num_heads, key_dim)
        
        # normailization layers 
        self.layer_norm_1 = tf.keras.layers.LayerNormalization()
        self.layer_norm_2 = tf.keras.layers.LayerNormalization()
       
    def call(self, embedded_articles):
        
        '''
        embedded_artiles: (batch_size (TBD), window_size (512), embedding_size (768))
        '''
        # self attention on embedded articles --> z (window_size x key_dims)
        z_matrix = self.self_atten(embedded_articles, embedded_articles)
        
        # add part of Add and Normalize 
        residuals = embedded_articles + z_matrix
        
        # normalize the added matrixes 
        normalized_resid = self.layer_norm_1(residuals)
        
        # feed forward the normalized output 
        ff_output = self.ff_layer(normalized_resid)
        
        # normalize the first normalization and the output of feed forward
        normalized_resid2 = normalized_resid + ff_output
        
        encoder_output = self.layer_norm_2(normalized_resid2)
        
        return encoder_output
    
    def get_config(self):
        return {'ff_layer': self.ff_layer, 'self_atten': self.self_atten, 
                'layer_norm_1': self.layer_norm_1, 'layer_norm_2': self.layer_norm_2}