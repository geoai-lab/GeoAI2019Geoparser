import numpy as np
import random
from keras.preprocessing.sequence import pad_sequences

NER_label = ["B-LOC","I-LOC"]

def readfile(filename):
    '''
    read file
    return format :
    [ ['EU', 'B-ORG'], ['rejects', 'O'], ['German', 'B-MISC'], ['call', 'O'], ['to', 'O'], ['boycott', 'O'], ['British', 'B-MISC'], ['lamb', 'O'], ['.', 'O'] ]
    '''
    f = open(filename)
    sentences = []
    sentence = []
    for line in f:
        if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
            if len(sentence) > 0:
                sentences.append(sentence)
                sentence = []
            continue
        splits = line.split(' ')

        if splits[-1].strip('\n') in NER_label:
            sentence.append([splits[0], splits[-1].strip('\n')])
        else:
            sentence.append([splits[0], 'O'])

    if len(sentence) > 0:
        sentences.append(sentence)
        sentence = []
    return sentences


def createBatches(data):
    l = []
    for i in data:
        l.append(len(i[0]))
    l = set(l)
    batches = []
    batch_len = []
    z = 0
    for i in l:
        for batch in data:
            if len(batch[0]) == i:
                batches.append(batch)
                z += 1
        batch_len.append(z)
    return batches, batch_len


def createMatrices_char(sentences, label2Idx):
    dataset = []

    wordCount = 0

    for sentence in sentences:

        labelIndices = []
        wordStrings = ""

        for word, label in sentence:
            wordCount += 1
            wordStrings = wordStrings + word + " "

            labelIndices.append(label2Idx[label])

        dataset.append([labelIndices, wordStrings[:-1]])

    return dataset


def iterate_minibatches_char(dataset,batch_len):
    start = 0
    for i in batch_len:
        tokens = []
        char = []
        labels = []
        words = []
        data = dataset[start:i]
        start = i
        for dt in data:
            l, word = dt
            l = np.expand_dims(l, -1)
            labels.append(l)
            words.append(word)

        yield np.asarray(labels), np.array(words, dtype=object)[:, np.newaxis]

