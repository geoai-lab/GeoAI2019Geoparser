import numpy as np
from keras.models import Model, load_model
from keras.layers import TimeDistributed, Dense, Embedding, Input, LSTM, Bidirectional, concatenate
from preprocess import readfile, createBatches, createMatrices_char, iterate_minibatches_char, addCharInformatioin,\
    padding
from keras.utils import Progbar
from keras.initializers import RandomUniform
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy
from ELMo import ElmoEmbeddingLayer

epochs = 50
Model_DIR = '/path/to/your/main/directory/'


def tag_dataset(dataset, model_path):
    model = load_model(model_path)
    correctLabels = []
    predLabels = []
    b = Progbar(len(dataset))
    for i, data in enumerate(dataset):
        tokens, labels = data
        tokens = np.asarray([tokens])
        # char = np.asarray([char])
        pred = model.predict([tokens], verbose=False)[0]
        pred = pred.argmax(axis=-1)  # Predict the classes
        correctLabels.append(labels)
        predLabels.append(pred)
        b.update(i)
    b.update(i + 1)
    return predLabels, correctLabels

#
trainSentences = readfile("/path/to/your/CoNLL-2003/format/training/dataset")

trainSentences = addCharInformatioin(trainSentences)

labelSet = set()
words = {}

for sentence in trainSentences:
    for token, char, label in sentence:
        labelSet.add(label)
        words[token.lower()] = True

# :: Create a mapping for the labels ::
label2Idx = {}
for label in labelSet:
    label2Idx[label] = len(label2Idx)


# :: Read in word embeddings ::
word2Idx = {}
wordEmbeddings = []

fEmbeddings = open("/path/to/your/WordEmbeddding/glove.6B/glove.6B.300d.txt", encoding="utf-8")

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


char2Idx = {"PADDING": 0, "UNKNOWN": 1}
for c in " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,-_()[]{}!?:;#'\"/\\%$`&=*+@^~|":
    char2Idx[c] = len(char2Idx)

train_set = padding(createMatrices_char(trainSentences, word2Idx, label2Idx, char2Idx))


idx2Label = {v: k for k, v in label2Idx.items()}

np.save("outputs/idx2Label.npy", idx2Label)
np.save("outputs/word2Idx.npy", word2Idx)
np.save("outputs/char2Idx.npy", char2Idx)

train_batch, train_batch_len = createBatches(train_set)

words_input = Input(shape=(None,), dtype='int32', name='words_input')
words = Embedding(input_dim=wordEmbeddings.shape[0], output_dim=wordEmbeddings.shape[1], weights=[wordEmbeddings],
                  trainable=False)(words_input)

character_input = Input(shape=(None, 52), name='char_input')
embed_char_out = TimeDistributed(Embedding(input_dim=len(char2Idx), output_dim=50, embeddings_initializer=
                                 RandomUniform(minval=-0.5, maxval=0.5)), name='char_embedding')(character_input)
char_lstm = TimeDistributed(Bidirectional(LSTM(25, return_sequences=False, return_state=False, recurrent_dropout=0.25)),
                            name='char_LSTM')(embed_char_out)

#
words_elmo_input = Input(shape=(1,),  dtype='string', name='words_elmo_input')
words_elmo = ElmoEmbeddingLayer(trainable=False)(words_elmo_input)


output = concatenate([words, char_lstm, words_elmo], axis=-1)

output_lstm = Bidirectional(LSTM(100, return_sequences=True, dropout=0.50, recurrent_dropout=0.25))(output)

# output_ffn = TimeDistributed(Dense(len(label2Idx), activation='softmax'))(output_lstm)

my_crf = CRF(len(label2Idx), sparse_target=True, name='CRF_layer')(output_lstm)

model = Model(inputs=[words_input, character_input, words_elmo_input], outputs=[my_crf])

# my_optimizer = optimizers.SGD(loss='sparse_categorical_crossentropy', optimizer='sgd', lr=0.15, decay=0.99)
# model.compile(loss='sparse_categorical_crossentropy', optimizer='nadam')

model.compile(optimizer='adam', loss=crf_loss, metrics=[crf_viterbi_accuracy])
model.summary()


for epoch in range(epochs):
    print("Epoch %d/%d" % (epoch, epochs))
    a = Progbar(len(train_batch_len))
    for i, batch in enumerate(iterate_minibatches_char(train_batch, train_batch_len)):
        labels, tokens, chars, strings = batch
        model.train_on_batch([tokens, chars, strings], labels)
        a.update(i)
    a.update(i + 1)
    print(' ')

model.save(Model_DIR + "/outputs/DM_NLP_model.h5")


#   Performance on test dataset
# print("Test the trained model on the testing dataset:")
# predLabels, correctLabels = tag_dataset(test_batch, str(Model_DIR + "model.h5"))
# pre_test, rec_test, f1_test = compute_f1(predLabels, correctLabels, idx2Label)
# print("Test-Data: Prec: %.3f, Rec: %.3f, F1: %.3f" % (pre_test, rec_test, f1_test))