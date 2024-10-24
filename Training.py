import tensorflow as tf
import numpy as np
import glob
import json
import random
from music21 import converter, instrument, note, chord

notes = []

for file in glob.glob("musiques/*.mid"):  
    midi= converter.parse(file)
    parts = instrument.partitionByInstrument(midi)
    if parts: 
        notes_to_parse = parts.parts[0].recurse()   
    else:  
        notes_to_parse = midi.flat.notes
       
    for element in notes_to_parse:   
        if isinstance(element, note.Note):
            notes.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            for n in element.notes :
                notes.append(str(n.pitch))

vocab=set(notes)
print(len(vocab))
vocab_to_int = {l:i for i,l in enumerate(vocab)}
int_to_vocab = {i:l for i,l in enumerate(vocab)}


with open("model_rnn_vocab_to_int", "w") as f:
    f.write(json.dumps(vocab_to_int))
with open("model_rnn_int_to_vocab", "w") as f:
    f.write(json.dumps(int_to_vocab))

encoded = [vocab_to_int[l] for l in notes]

inputs, targets = encoded, encoded[1:]

def gen_batch(inputs, targets, seq_len, batch_size, noise=0):
        Lbatch=[]
        chunk_size=len(inputs) // batch_size
        seq_per_chunk=chunk_size // seq_len
        
        
        for s in range(seq_per_chunk):
            batch_inputs=np.zeros((batch_size, seq_len))
            batch_targets=np.zeros((batch_size, seq_len))
            
            for k in range(batch_size):
                deb=s*seq_len+k*chunk_size
                fin=deb+seq_len
                batch_inputs[k]=inputs[deb:fin]
                batch_targets[k]=targets[deb+1:fin+1]
                if noise > 0:                          
                    noise_indices = np.random.choice(seq_len, noise)
                    batch_inputs[k][noise_indices] = np.random.randint(0, vocab_size)
            Lbatch.append((batch_inputs,batch_targets))
            
        return Lbatch

#OnehotEncoding Layer
class OneHot(tf.keras.layers.Layer):
    def __init__(self, depth, **kwargs):
        super(OneHot, self).__init__(**kwargs)
        self.depth = depth

    def call(self, x, mask=None):
        return tf.one_hot(tf.cast(x, tf.int32), self.depth)


    def get_config(self):
        # For serialization with 'custom_objects'
        config = super().get_config()
        config['depth'] = self.depth
        
        return config

    
    
vocab_size = len(vocab)

tf_inputs = tf.keras.Input(shape=(None,), batch_size=90)

one_hot = OneHot(len(vocab))(tf_inputs)

rnn_layer1 = tf.keras.layers.LSTM(128, return_sequences=True, stateful=True)(one_hot)
rnn_layer2 = tf.keras.layers.LSTM(128, return_sequences=True, stateful=True)(rnn_layer1)

hidden_layer = tf.keras.layers.Dense(128, activation="relu")(rnn_layer2)
outputs = tf.keras.layers.Dense(vocab_size, activation="softmax")(hidden_layer)

model = tf.keras.Model(inputs=tf_inputs, outputs=outputs)


loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(lr=0.001)

train_loss = tf.keras.metrics.Mean(name='train_loss')
# Accuracy
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

@tf.function
def train_step(inputs, targets):
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss = loss_object(targets, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    train_loss(loss)
    train_accuracy(targets, predictions)

model.reset_states()

for epoch in range(5000):
    for batch_inputs, batch_targets in gen_batch(inputs, targets, 100, 90 , noise=13):
        train_step(batch_inputs, batch_targets)
    template = '\r Epoch {}, Train Loss: {}, Train Accuracy: {}'
    print(template.format(epoch, train_loss.result(), train_accuracy.result()*100), end="")
    model.reset_states()


model.save("model_rnn.h5")