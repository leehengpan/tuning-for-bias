import tensorflow as tf

# Decoder block for the transformer model 
class DecoderBlock(tf.keras.layers.Layer):
    def __init__(self, emb_sz, num_heads, key_dim, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)

        # dense layers for decoder block --> may need to change output 768 for embedding size 
        self.ff_layer = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(emb_sz)
        ])
        
        # self attention layer 
        self.self_atten = tf.keras.layers.MultiHeadAttention(num_heads, key_dim)
        
        # cross attention layer 
        self.cross_atten = tf.keras.layers.MultiHeadAttention(num_heads, key_dim)
        
        # normailization layers 
        self.layer_norm_1 = tf.keras.layers.LayerNormalization()
        self.layer_norm_2 = tf.keras.layers.LayerNormalization()
        self.layer_norm_3 = tf.keras.layers.LayerNormalization()
       

    def call(self, encoder_output, decoder_input):
        
        '''
        encoder_output: (batch_size (TBD), window_size (512), embedding_size (768))
        '''
        
        # self atten on the inputs to the decoder --> titles 
        z_matrix = self.self_atten(decoder_input, decoder_input, use_causal_mask=True)
        
        # add and normalize the residuals from the self atten mechanism 
        residuals = decoder_input + z_matrix
        normalized_resid = self.layer_norm_1(residuals)
        
        # perform cross attention on normalized self-atten and the decoder context 
        cross_atten_matrix = self.cross_atten(normalized_resid, encoder_output)
        
        # normalize the first normalization and the output of feed forward
        residual_2 = normalized_resid + cross_atten_matrix
        normalized_resid2 = self.layer_norm_2(residual_2)
        
        # feed forward the normalized output 
        ff_output = self.ff_layer(normalized_resid2)
        
        # normalize and add the second layers 
        residual_3 = ff_output + normalized_resid2
        decoder_output = self.layer_norm_3(residual_3)
        
        return decoder_output
    
    def get_config(self):
        return {'ff_layer': self.ff_layer, 'self_atten': self.self_atten,
                'cross_atten': self.cross_atten, 'layer_norm_1': self.layer_norm_1,
                'layer_norm_2': self.layer_norm_2, 'layer_norm_3': self.layer_norm_3}