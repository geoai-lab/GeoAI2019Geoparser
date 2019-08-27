import os
import numpy as np
from attention import Attention
from keras.layers import LSTM, Bidirectional, Embedding, TimeDistributed, Dense, Input
from keras.models import Model
from keras.utils import Progbar
from preprocess import readfile, iterate_minibatches, createBatches, createMatrices
from ELMo import ElmoEmbeddingLayer
import time

BASE_DIR = '/home/jiminwan/GeoAI2019/'
GLOVE_DIR = os.path.join(BASE_DIR, 'word_embedding/glove.6B/glove.6B.300d.txt')
MODEL_DIR = os.path.join(BASE_DIR, 'attention_biLSTM/outputs/attention_model.h5')
MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2
epochs = 50

## Read files
trainSentences = readfile("/home/jiminwan/GeoAI2019/trainingdata/CoNLL2003-train.txt")
# testSentences = readfile("/Users/jiminwan/Data/RNN_geoparser/trainingdata/CoNLL2003-test.txt")

## Build vocabulary dictionary

labelSet = set()
words = {}


for sentence in trainSentences:
    for token, label in sentence:
        labelSet.add(label)
        words[token.lower()] = True

# :: Create a mapping for the labels ::
label2Idx = {}
for label in labelSet:
    label2Idx[label] = len(label2Idx)

# Build index mapping words in the embeddings set to their embedding vector

print('Indexing word vectors.')

word2Idx = {}
wordEmbeddings = []

fEmbeddings = open(GLOVE_DIR, encoding="utf-8")

for line in fEmbeddings:
    split = line.strip().split(" ")
    word = split[0]

    if len(word2Idx) == 0:  # Add padding+unknown
        word2Idx["PADDING_TOKEN"] = len(word2Idx)
        vector = np.zeros(len(split) - 1)  # Zero vector vor 'PADDING' word
        wordEmbeddings.append(vector)

        word2Idx["UNKNOWN_TOKEN"] = len(word2Idx)
        vector = np.random.uniform(-0.25, 0.25, len(split) - 1)
        wordEmbeddings.append(vector)

    if split[0].lower() in words:
        vector = np.array([float(num) for num in split[1:]])
        wordEmbeddings.append(vector)
        word2Idx[split[0]] = len(word2Idx)

wordEmbeddings = np.array(wordEmbeddings)


# second, prepare CoNLL format training and validation dataset
print('Processing the training dataset')

train_set = createMatrices(trainSentences, word2Idx, label2Idx)
idx2Label = {v: k for k, v in label2Idx.items()}
print(idx2Label)
train_batch, train_batch_len = createBatches(train_set)

np.save("outputs/idx2Label.npy", idx2Label)
np.save("outputs/word2Idx.npy", word2Idx)

## Build the LSTM model

### Word input
words_input_int = Input(shape=(None,), dtype='int32', name='words_input1')

### Word Embedding layer
words = Embedding(input_dim=wordEmbeddings.shape[0], output_dim=wordEmbeddings.shape[1], weights=[wordEmbeddings],
               trainable=False)(words_input_int)
print(words._keras_shape)

### Bi-LSTM layer
output_lstm = Bidirectional(LSTM(100, return_sequences=True, dropout=0.50, recurrent_dropout=0.25))(words_elmo)

### Self attention layer
output_atten = Attention(8, 16)([output_lstm, output_lstm, output_lstm])

### FFN layer
output_ffn1 = TimeDistributed(Dense(300, activation = 'relu'))(output_atten)
# output_ffn1 = BatchNormalization()(output_ffn1)

output_ffn2 = TimeDistributed(Dense(300, activation = 'relu'))(output_ffn1)
# output_ffn2 = BatchNormalization()(output_ffn2)

output_ffn3 = TimeDistributed(Dense(300, activation = 'relu'))(output_ffn2)

### Output layer using softmax
output_ffn = TimeDistributed(Dense(len(idx2Label), activation='softmax'))(output_ffn3)

### Compile the whole model
model = Model(inputs=[words_elmo_input], outputs=[output_ffn])
model.compile(loss='sparse_categorical_crossentropy', optimizer='nadam')
model.summary()


# Train the model

for epoch in range(epochs):
    print("Epoch %d/%d" % (epoch, epochs))
    a = Progbar(len(train_batch_len))
    for i, batch in enumerate(iterate_minibatches(train_batch, train_batch_len)):
        labels, tokens = batch
        model.train_on_batch([tokens], labels)
        a.update(i)
    a.update(i + 1)
    print(' ')

model.save(MODEL_DIR)






