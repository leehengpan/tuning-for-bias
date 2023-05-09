import tensorflow as tf
from encoder import EncoderBlock
from decoder import DecoderBlock

# model with multiple encoder blocks and multiple decoder blocks
class TransformerModel(tf.keras.Model):
    
    def __init__(self, emb_sz, num_heads, key_dim, vocab_size, **kwargs):
        
        super().__init__()
        
        # create encoder and decoder blocks from classes 
        self.encoder_block1 = EncoderBlock(emb_sz,num_heads,key_dim)
        self.encoder_block2 = EncoderBlock(emb_sz,num_heads,key_dim)
        # self.encoder_block3 = EncoderBlock(emb_sz,num_heads,key_dim)
        # self.encoder_block4 = EncoderBlock(emb_sz,num_heads,key_dim)
        # self.encoder_block5 = EncoderBlock(emb_sz,num_heads,key_dim)
        # self.encoder_block6 = EncoderBlock(emb_sz,num_heads,key_dim)
        # self.encoder_block7 = EncoderBlock(emb_sz,num_heads,key_dim)
        # self.encoder_block8 = EncoderBlock(emb_sz,num_heads,key_dim)
     
        self.decoder_block1 = DecoderBlock(emb_sz,num_heads,key_dim)
        self.decoder_block2 = DecoderBlock(emb_sz,num_heads,key_dim)
        # self.decoder_block3 = DecoderBlock(emb_sz,num_heads,key_dim)
        # self.decoder_block4 = DecoderBlock(emb_sz,num_heads,key_dim)
        # self.decoder_block5 = DecoderBlock(emb_sz,num_heads,key_dim)
        # self.decoder_block6 = DecoderBlock(emb_sz,num_heads,key_dim)
        # self.decoder_block7 = DecoderBlock(emb_sz,num_heads,key_dim)
        # self.decoder_block8 = DecoderBlock(emb_sz,num_heads,key_dim)
    
        # dense layer for final output 
        self.dense_layer = tf.keras.layers.Dense(vocab_size, activation='softmax')

    
    def call(self, inputs):

        # encoder blocks 
        encoder_output = self.encoder_block1(inputs[0])
        encoder_output = self.encoder_block2(encoder_output)
        # encoder_output = self.encoder_block3(encoder_output)
        # encoder_output = self.encoder_block4(encoder_output)
        # encoder_output = self.encoder_block5(encoder_output)
        # encoder_output = self.encoder_block6(encoder_output)
        # encoder_output = self.encoder_block7(encoder_output)
        # encoder_output = self.encoder_block8(encoder_output)

        # decoder blocks 
        decoder_output = self.decoder_block1(encoder_output, inputs[1])
        decoder_output = self.decoder_block2(encoder_output, decoder_output)
        # decoder_output = self.decoder_block3(encoder_output, decoder_output)
        # decoder_output = self.decoder_block4(encoder_output, decoder_output)
        # decoder_output = self.decoder_block5(encoder_output, decoder_output)
        # decoder_output = self.decoder_block6(encoder_output, decoder_output)
        # decoder_output = self.decoder_block7(encoder_output, decoder_output)
        # decoder_output = self.decoder_block8(encoder_output, decoder_output)

        # final logit outputs 
        logits = self.dense_layer(decoder_output)
        
        
        return logits
    
    def masked_loss(label, pred):
        mask = label != 0
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none')
        loss = tf.expand_dims(loss_object(label, pred), axis=2)

        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask

        loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
        return loss

    def masked_accuracy(label, pred):
        pred = tf.expand_dims(tf.argmax(pred, axis=2), axis=2)
        label = tf.cast(label, pred.dtype)
        match = label == pred

        mask = label != 0

        match = match & mask

        match = tf.cast(match, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        return tf.reduce_sum(match)/tf.reduce_sum(mask)
    
    def get_config(self):
        return {'encoder_block1': self.encoder_block1,'encoder_block2': self.encoder_block2, 
                'decoder_block1': self.decoder_block1, 'decoder_block2': self.decoder_block2, 
                'dense_layer': self.dense_layer}
