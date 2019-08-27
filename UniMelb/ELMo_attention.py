import os
import numpy as np
from ELMo import ElmoEmbeddingLayer
from attention import Attention
from keras.layers import LSTM, Bidirectional, TimeDistributed, Dense, Input, BatchNormalization
from keras.models import Model
from keras.utils import Progbar
from keras import optimizers
from preprocess_ELMo import readfile, iterate_minibatches_char, createBatches, createMatrices_char

BASE_DIR = '/home/jiminwan/GeoAI2019'
GLOVE_DIR = os.path.join(BASE_DIR, 'word_embedding/glove.6B/glove.6B.300d.txt')
MODEL_DIR = os.path.join(BASE_DIR, 'attention_biLSTM/outputs/attention_elmo_model.h5')

VALIDATION_SPLIT = 0.2
epochs = 50
decay_rate = 10**(-9)

## Read files
trainSentences = readfile("/home/jiminwan/GeoAI2019/trainingdata/CoNLL2003-train.txt")

## Build vocabulary dictionary
labelSet = set()

for sentence in trainSentences:
    for token, label in sentence:
        labelSet.add(label)


# :: Create a mapping for the labels ::
label2Idx = {}
for label in labelSet:
    label2Idx[label] = len(label2Idx)


# second, prepare CoNLL format training and validation dataset
print('Processing the training dataset')

train_set = createMatrices_char(trainSentences, label2Idx)
idx2Label = {v: k for k, v in label2Idx.items()}
print(idx2Label)
train_batch, train_batch_len = createBatches(train_set)

np.save("outputs/idx2Label.npy", idx2Label)
# np.save("models/word2Idx.npy", word2Idx)

## Build the LSTM model

### Word input

words_elmo_input = Input(shape=(1,),  dtype='string', name='words_elmo_input')
words_elmo = ElmoEmbeddingLayer(trainable=False)(words_elmo_input)


### Bi-LSTM layer
output_lstm = Bidirectional(LSTM(300, return_sequences=True, dropout=0.50, recurrent_dropout=0.25))(words_elmo)

### Self attention layer
output_atten = Attention(8, 16)([output_lstm, output_lstm, output_lstm])

### FFN layer
output_ffn1 = TimeDistributed(Dense(300, activation = 'relu'))(output_atten)
# output_ffn1 = BatchNormalization()(output_ffn1)

output_ffn2 = TimeDistributed(Dense(300, activation = 'relu'))(output_ffn1)
# output_ffn2 = BatchNormalization()(output_ffn2)

output_ffn3 = TimeDistributed(Dense(300, activation = 'relu'))(output_ffn2)
# output_ffn3 = BatchNormalization()(output_ffn3)

### Output layer using softmax

output_softmax = TimeDistributed(Dense(len(label2Idx), activation='softmax'))(output_ffn3)

### Compile the whole model
model = Model(inputs=[words_elmo_input], outputs=[output_softmax])
adam_opt = optimizers.adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=decay_rate, amsgrad=False)
model.compile(loss='sparse_categorical_crossentropy', optimizer=adam_opt)
model.summary()


## Train the model

for epoch in range(epochs):
    print("Epoch %d/%d" % (epoch, epochs))
    a = Progbar(len(train_batch_len))
    for i, batch in enumerate(iterate_minibatches_char(train_batch, train_batch_len)):
        labels, strings = batch
        model.train_on_batch([strings], labels)
        a.update(i)
    a.update(i + 1)
    print(' ')

model.save(MODEL_DIR)






