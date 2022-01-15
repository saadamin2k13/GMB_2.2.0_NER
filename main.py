import os
import csv
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from glob import glob

import warnings
warnings.filterwarnings("ignore")


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import *

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


files = glob("gmb-2.2.0/data/*/*/*.tags")

len(files)

print(files[:2])

ner_tags = Counter()
iob_tags = Counter()


def strip_ner_subcat(tag):
    # NER tag from {cat}-{subcat}
    # we only want first part
    return tag.split("-")[0]


## take a sequence of tags and convert them into IOB format

def iob_format(ners):
    iob_tokens = []
    for i, token in enumerate(ners):
        if token != 'O':  # other
            if i == 0:
                token = "B-" + token  # start of sentence tag
            elif ners[i - 1] == token:
                token = "I-" + token  # continues
            else:
                token = "B-" + token
        iob_tokens.append(token)
        iob_tags[token] += 1
    return iob_tokens


total_sentences = 0
outfiles = []

for i, file in enumerate(files):
    with open(file, 'rb') as content:
        data = content.read().decode('utf-8').strip()
        sentences = data.split("\n\n")
        #         print(i, file, len(sentences))
        total_sentences += len(sentences)

        with open("./ner/" + str(i) + "-" + os.path.basename(file), 'w') as outfile:
            outfiles.append("./ner/" + str(i) + "-" + os.path.basename(file))
            writer = csv.writer(outfile)
            for sentence in sentences:
                toks = sentence.split("\n")
                words, pos, ner = [], [], []

                for tok in toks:
                    t = tok.split("\t")
                    words.append(t[0])
                    pos.append(t[1])
                    ner_tags[t[3]] += 1
                    ner.append(strip_ner_subcat(t[3]))
                writer.writerow([" ".join(words),
                                 " ".join(iob_format(ner)),
                                 " ".join(pos)])


print("Total number of sentences:", total_sentences)

#normalizing and vectoring

files = glob("./ner/*.tags")

data_pd = pd.concat([pd.read_csv(f, header=None, names=['text', 'label', 'pos']) for f in files], ignore_index=True)

data_pd.info()


print(data_pd.head())

text_tok = Tokenizer(filters='[\\]^\t\n', lower=False, split=' ', oov_token='<OOV>')
pos_tok = Tokenizer(filters='\t\n', lower=False, split=' ', oov_token='<OOV>')
ner_tok = Tokenizer(filters='\t\n', lower=False, split=' ', oov_token='<OOV>')

text_tok.fit_on_texts(data_pd['text'])
pos_tok.fit_on_texts(data_pd['pos'])
ner_tok.fit_on_texts(data_pd['label'])

text_config = text_tok.get_config()
ner_config = ner_tok.get_config()
print(ner_config)

text_vocab = eval(text_config['index_word'])
ner_vocab = eval(ner_config['index_word'])

print("Unique words in vocab:", len(text_vocab))
print("Unique NER tags in vocab:", len(ner_vocab))

x_tok = text_tok.texts_to_sequences(data_pd['text'])
y_tok = ner_tok.texts_to_sequences(data_pd['label'])

# pad sequences to a max length

max_len = 50

x_pad = pad_sequences(x_tok, padding='post', maxlen=max_len)
y_pad = pad_sequences(y_tok, padding='post', maxlen=max_len)

print(x_pad.shape, y_pad.shape)

num_classes = len(ner_vocab) + 1
Y = keras.utils.to_categorical(y_pad, num_classes=num_classes)

print(Y.shape)

vocab_size = len(text_vocab) + 1
embedding_dims = 64
rnn_units = 100

BATCH_SIZE = 128

model = Sequential([
    Embedding(vocab_size, embedding_dims, mask_zero=True, input_length=50),
    Bidirectional(LSTM(rnn_units, return_sequences=True, dropout=0.2, kernel_initializer="he_normal")),
    TimeDistributed(Dense(rnn_units, "relu")),
    Dense(num_classes, 'softmax')
])

print(model.summary())

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

history = model.fit(x_pad, Y, batch_size=BATCH_SIZE, epochs=100, validation_split=0.2)

#!pip install tensorflow_addons

from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
import tensorflow_addons as tfa


class CRFLayer(Layer):
    """
    Computes the log likelihood during training
    Performs Viterbi decoding during predictions
    """

    def __init__(self, label_size, mask_id=0,
                 trans_params=None, name='crf',
                 **kwargs):
        super(CRFLayer, self).__init__(name=name, **kwargs)
        self.label_size = label_size
        self.mask_id = mask_id
        self.transition_params = None

        if trans_params is None:  # not reloading pretrained params
            self.transition_params = tf.Variable(tf.random.uniform(shape=(label_size, label_size)), trainable=False)
        else:
            self.transition_params = trans_params

    def call(self, inputs, seq_lengths, training=None):
        if training is None:
            training = K.learning_phase()

        # during training, this layer just returns the logits
        if training:
            return inputs

        # viterbi decode logic to proper
        # results at inference
        _, max_seq_len, _ = inputs.shape
        seqlens = seq_lengths
        paths = []
        for logit, text_len in zip(inputs, seqlens):
            viterbi_path, _ = tfa.text.viterbi_decode(logit[:text_len], self.transition_params)
            paths.append(self.pad_viterbi(viterbi_path, max_seq_len))

        return tf.convert_to_tensor(paths)

    def pad_viterbi(self, viterbi, max_seq_len):
        if len(viterbi) < max_seq_len:
            viterbi = viterbi + [self.mask_id] * (max_seq_len - len(viterbi))
        return viterbi

    def loss(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(self.get_proper_labels(y_true), y_pred.dtype)

        seq_lengths = self.get_seq_lengths(y_true)
        log_likelihoods, self.transition_params = tfa.text.crf_log_likelihood(y_pred, y_true, seq_lengths)

        # save transition params
        self.transition_params = tf.Variable(self.transition_params, trainable=False)
        # calc loss
        loss = - tf.reduce_mean(log_likelihoods)
        return loss

    def get_proper_labels(self, y_true):
        shape = y_true.shape
        if len(shape) > 2:
            return tf.argmax(y_true, -1, output_type=tf.int32)
        return y_true

    def get_seq_lengths(self, matrix):
        # matrix is of shape (batch_size, max_seq_len)
        mask = tf.not_equal(matrix, self.mask_id)
        seq_lengths = tf.math.reduce_sum(tf.cast(mask, dtype=tf.int32), axis=-1)

        return seq_lengths


class NerModel(tf.keras.Model):
    def __init__(self, hidden_num, vocab_size,
                 label_size, embedding_size,
                 name='BilstmCrfModel', **kwargs):
        super(NerModel, self).__init__(name=name, **kwargs)
        self.vocab_size = vocab_size
        self.label_size = label_size
        self.embedding = Embedding(vocab_size, embedding_size, mask_zero=True, name="embedding")
        self.biLSTM = Bidirectional(LSTM(hidden_num, return_sequences=True), name='bilstm')
        self.dense = TimeDistributed(Dense(label_size), name='dense')
        self.crf = CRFLayer(self.label_size, name='crf')

    def call(self, text, labels=None, training=None):
        seq_lengths = tf.math.reduce_sum(
            tf.cast(tf.math.not_equal(text, 0), dtype=tf.int32), axis=-1)

        if training is None:
            training = K.learning_phase()

        inputs = self.embedding(text)
        bilstm = self.biLSTM(inputs)
        logits = self.dense(bilstm)
        outputs = self.crf(logits, seq_lengths, training)

        return outputs


vocab_size = len(text_vocab) + 1
embedding_dims = 64
rnn_units = 100
batch_size = 90

num_classes = len(ner_vocab) + 1

blc_model = NerModel(rnn_units, vocab_size, num_classes, embedding_dims, dynamic=True)
optimizer = keras.optimizers.Adam(learning_rate=1e-3)


total_sentences = 62010
test_size = round(total_sentences / batch_size * 0.2)

X_train = x_pad[batch_size*test_size:]
Y_train = Y[batch_size*test_size:]

X_test = x_pad[:batch_size*test_size]
Y_test = Y[:batch_size*test_size]

Y_train_int = tf.cast(Y_train, dtype=tf.int32)
Y_test_int = tf.cast(Y_test, dtype=tf.int32)

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train_int))
train_dataset = train_dataset.batch(batch_size, drop_remainder=True)

## training custom model

loss_metric = keras.metrics.Mean()

epochs = 100

# iterate over epochs
for epoch in range(epochs):
    print(f'Start of epoch {epoch + 1},')

    # iterate over batches of the dataset
    for step, (text_batch, labels_batch) in enumerate(train_dataset):
        labels_max = tf.argmax(labels_batch, -1, output_type=tf.int32)

        with tf.GradientTape() as tape:
            logits = blc_model(text_batch, training=True)
            loss = blc_model.crf.loss(labels_max, logits)
            grads = tape.gradient(loss, blc_model.trainable_weights)
            optimizer.apply_gradients(zip(grads, blc_model.trainable_weights))

            loss_metric(loss)

            if step % 50 == 0:
                print(f"step {step + 1}: mean loss = {loss_metric.result()}")



print(blc_model.summary())

test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test_int))
test_dataset = test_dataset.batch(batch_size, drop_remainder=True)

out = blc_model.predict(test_dataset.take(1))

print("Ground Truth:", ner_tok.sequences_to_texts([tf.argmax(Y_test[3],-1).numpy()]))
print("\nPrediction:", ner_tok.sequences_to_texts([out[3]]))


# custom accuracy

def np_precision(pred, true):
    # expect numpy arrays
    assert pred.shape == true.shape
    assert len(pred.shape) == 2
    mask_pred = np.ma.masked_equal(pred, 0)
    mask_true = np.ma.masked_equal(true, 0)
    acc = np.equal(mask_pred, mask_true)

    return np.mean(acc.compressed().astype('int'))

print(np_precision(out, tf.argmax(Y_test[:batch_size], -1).numpy()))