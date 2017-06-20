"""
by Yuanzhi Ke. June 2017
require keras 1.2.2, tensorflow 1.1, numpy
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import os, re
import pandas as pd
from bs4 import BeautifulSoup
import json
from keras import backend as K

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Dense, Input
from keras.layers import Embedding, Merge, Dropout, GRU, Bidirectional, TimeDistributed
from keras.layers.merge import Dot
from keras.models import Model

from mulembedding import MulEmbedding
from attention import AttentionWithContext
from embeddingdot import EmbeddingDot

from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

flags = tf.app.flags

flags.DEFINE_string("save_path", "/tmp/", "Directory to write the model.")
flags.DEFINE_string(
    "train_data", "Corpus/text8",
    "Training data. E.g., unzipped file http://mattmahoney.net/dc/text8.zip.")
flags.DEFINE_string(
    "eval_data", None, "Analogy questions. "
                       "See README.md for how to get 'questions-words.txt'.")
flags.DEFINE_string(
    "pretrained_emb", None,
    "Pretrained single prototype word embedding data."
)
flags.DEFINE_integer("embedding_size", 100, "The embedding dimension size.")
flags.DEFINE_integer(
    "epochs_to_train", 15,
    "Number of epochs to train. Each epoch processes the training data once "
    "completely.")
flags.DEFINE_float("learning_rate", 0.2, "Initial learning rate.")
flags.DEFINE_integer("num_neg_samples", 100,
                     "Negative samples per training example.")
flags.DEFINE_integer("batch_size", 16,
                     "Numbers of training examples each step processes."
                     "(no minibatching). Not used for classification task.")
flags.DEFINE_integer("concurrent_steps", 12,
                     "The number of concurrent training steps.")
flags.DEFINE_integer("window_size", 5,
                     "The number of words to predict to the left and right "
                     "of the target word.")
flags.DEFINE_integer("min_count", 5,
                     "The minimum number of word occurrences for it to be "
                     "included in the vocabulary.")
flags.DEFINE_integer("prototypes", 20,
                     "The number of prototypes of each word.")
flags.DEFINE_float("subsample", 1e-4,
                   "Subsample threshold for word occurrence. Words that appear "
                   "with higher frequency will be randomly down-sampled. Set "
                   "to 0 to disable.")
flags.DEFINE_integer("context_size", 10,
                     "The context, size. If you change it, you"
                     "Need to also change it in embeddings/word2vec_kernels.cc now.")

FLAGS = flags.FLAGS


class Options(object):
    def __init__(self):
        self.emb_dim = FLAGS.embedding_size
        self.train_data = FLAGS.train_data
        self.num_samples = FLAGS.num_neg_samples
        self.learning_rate = FLAGS.learning_rate
        self.epochs_to_train = FLAGS.epochs_to_train
        self.concurrent_steps = FLAGS.concurrent_steps
        self.batch_size = FLAGS.batch_size
        self.window_size = FLAGS.window_size
        self.min_count = FLAGS.min_count
        self.subsample = FLAGS.subsample
        self.save_path = FLAGS.save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.eval_data = FLAGS.eval_data
        self.prototypes = FLAGS.prototypes
        self.pretrained_emb = FLAGS.pretrained_emb
        self.context_size = FLAGS.context_size
        self.MAX_SENT_LENGTH = 100
        self.MAX_SENTS = 15
        self.MAX_NUM_WORDS = 20000  # Maximum number of words to work with
        # (if set, tokenization will be restricted to the top
        # num_words most common words in the dataset).
        self.VALIDATION_SPLIT = 0.2


def data_reader_ts(opts):
    # return tensorflow tensors
    word2vec = tf.load_op_library(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), 'embedding/word2vec_ops.so'))
    with tf.Graph().as_default():
        with tf.device("/cpu:0"):
            (words, counts, words_per_epoch, _epoch, _words, examples,
             labels, contexts) = word2vec.full_context_skipgram_word2vec(filename=opts.train_data,
                                                                         batch_size=opts.batch_size,
                                                                         window_size=opts.window_size,
                                                                         min_count=opts.min_count,
                                                                         subsample=opts.subsample)
            # (vocab) = \
            #     session.run([words])
    return (words, counts, words_per_epoch, _epoch, _words, examples,
            labels, contexts)


def build_final_embedding(opts, word_inputs, context_inputs):
    vocab = opts.vocab
    # create single prototype embedding layer
    if opts.pretrained_emb is None:
        single_embedding_layer = Embedding(input_dim=opts.vocab_size,
                                           output_dim=opts.emb_dim,
                                           input_length=opts.context_size,
                                           trainable=True)
    else:
        single_embeddings_index = {}
        f = open(opts.pretrained_emb)
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            single_embeddings_index[word] = coefs
        f.close()

        print('Found %s pre-trained word vectors.' % len(single_embeddings_index))

        single_embedding_matrix = np.zeros((len(vocab) + 1, opts.emb_dim))
        for i, word in enumerate(vocab):
            embedding_vector = single_embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                single_embedding_matrix[i] = embedding_vector

        single_embedding_layer = Embedding(opts.vocab_size,
                                           opts.emb_dim,
                                           weights=[single_embedding_matrix],
                                           input_length=opts.context_size,
                                           trainable=True)

    # create multiple prototype embedding layer
    multiple_embedding_layer = MulEmbedding(input_dim=opts.vocab_size,
                                            output_prototypes=opts.prototypes,
                                            output_dim=opts.emb_dim,
                                            is_word_input=True,
                                            trainable=True)

    # [batch_size, prototypes, emb_dim]
    multiple_embeded = multiple_embedding_layer(word_inputs)
    embedded_context = single_embedding_layer(context_inputs)
    l_gru = Bidirectional(GRU(opts.emb_dim, return_sequences=True))(embedded_context)
    # [batch_size, prototypes]
    l_dense = TimeDistributed(Dense(opts.prototypes))(l_gru)
    l_att = AttentionWithContext()(l_dense)
    #
    # get final embedding
    embedding_dot = Dot(axes=[1, 1])
    final_embedding = embedding_dot([multiple_embeded, l_att])

    return final_embedding


def build_final_embedding_from_sentence(opts, sentence_input):
    vocab = opts.vocab
    # create single prototype embedding layer
    if opts.pretrained_emb is None:
        single_embedding_layer = Embedding(input_dim=opts.vocab_size,
                                           output_dim=opts.emb_dim,
                                           input_length=opts.MAX_SENT_LENGTH,
                                           trainable=True)
    else:
        single_embeddings_index = {}
        f = open(opts.pretrained_emb)
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            single_embeddings_index[word] = coefs
        f.close()

        print('Found %s pre-trained word vectors.' % len(single_embeddings_index))

        single_embedding_matrix = np.zeros((len(vocab) + 1, opts.emb_dim))
        for i, word in enumerate(vocab):
            embedding_vector = single_embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                single_embedding_matrix[i] = embedding_vector

        single_embedding_layer = Embedding(opts.vocab_size,
                                           opts.emb_dim,
                                           weights=[single_embedding_matrix],
                                           input_length=opts.MAX_SENT_LENGTH,
                                           trainable=True)

    # create multiple prototype embedding layer
    multiple_embedding_layer = MulEmbedding(input_dim=opts.vocab_size,
                                            output_prototypes=opts.prototypes,
                                            output_dim=opts.emb_dim,
                                            is_word_input=False,
                                            trainable=True)

    multiple_embeded = multiple_embedding_layer(sentence_input)  # [batch_size, sentence_len, prototypes, emb]
    embedded_context = single_embedding_layer(sentence_input)
    l_gru_emb = Bidirectional(GRU(opts.emb_dim, return_sequences=True))(embedded_context)  # [batch_size, sentence_len, emb*2]
    l_dense_emb = TimeDistributed(Dense(opts.prototypes))(l_gru_emb)  # [batch_size, sentence_len, protptyes]
    l_att_emb = AttentionWithContext()(l_dense_emb)  #[batch_size, prototypes]

    # Note: The following has moved into embeddingdot. June 20, 2017
    # repeat l_att for broadcasting because tf cannot automatically do it for this case
    # l_att_tile = K.tile(l_att, [1, opts.MAX_SENT_LENGTH])
    # l_att= K.reshape(l_att_tile,
    #                    [tf.shape(l_att)[0],
    #                     opts.MAX_SENT_LENGTH, opts.prototypes])
    #
    # get final embedding
    embedding_dot = EmbeddingDot(axes=[1, 2], sen_len=opts.MAX_SENT_LENGTH,
                                 proto=opts.prototypes)
    final_embedding = embedding_dot([l_att_emb, multiple_embeded])

    return final_embedding


def build_nce_w2v_forward(opts, embedding, word_inputs, label_inputs):
    """
    Build the forword nn to train word embeddings
    :param opts: Option instantce
    :param embedding: Output tensor of Final embedding layer
    :param word_inputs: placeholder or Input layer for word inputs
    :param label_inputs: placeholder or Inpu layer for label inputs
    :return: true_logits, sampled_logits
    """
    # Softmax weight: [vocab_size, emb_dim]. Transposed.
    sm_w_t = tf.Variable(
        tf.zeros([opts.vocab_size, opts.emb_dim]),
        name="sm_w_t")

    # Softmax bias: [vocab_size].
    sm_b = tf.Variable(tf.zeros([opts.vocab_size]), name="sm_b")

    # Nodes to compute the nce loss w/ candidate sampling.
    labels_matrix = tf.reshape(
        tf.cast(label_inputs,
                dtype=tf.int64),
        [opts.batch_size, 1])

    # Negative sampling.
    sampled_ids, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
        true_classes=labels_matrix,
        num_true=1,
        num_sampled=opts.num_samples,
        unique=True,
        range_max=opts.vocab_size,
        distortion=0.75,
        unigrams=opts.vocab_counts.tolist()))

    # Weights for labels: [batch_size, emb_dim]
    true_w = tf.nn.embedding_lookup(sm_w_t, label_inputs)
    # Biases for labels: [batch_size, 1]
    true_b = tf.nn.embedding_lookup(sm_b, label_inputs)

    # Weights for sampled ids: [num_sampled, emb_dim]
    sampled_w = tf.nn.embedding_lookup(sm_w_t, sampled_ids)
    # Biases for sampled ids: [num_sampled, 1]
    sampled_b = tf.nn.embedding_lookup(sm_b, sampled_ids)

    # True logits: [batch_size, 1]
    true_logits = tf.reduce_sum(tf.multiply(embedding, true_w), 1) + true_b

    # Sampled logits: [batch_size, num_sampled]
    # We replicate sampled noise labels for all examples in the batch
    # using the matmul.
    sampled_b_vec = tf.reshape(sampled_b, [opts.num_samples])
    sampled_logits = tf.matmul(embedding,
                               sampled_w,
                               transpose_b=True) + sampled_b_vec
    return true_logits, sampled_logits


def train_language_model(opts):
    # Global step: scalar, i.e., shape [].
    global_step = tf.Variable(0, name="global_step")
    # TODO


def build_HATT_RNN(opts, embedding, sentence_input, review_input):
    # TODO build HATT model
    pass


def word_embedding_task():
    opts = Options()
    session = tf.Session()

    (words, counts, words_per_epoch, _epoch, _words, examples,
     labels, contexts) = data_reader_ts(opts)

    (opts.vocab) = session.run([words])

    opts.vocab_size = len(opts.vocab)
    _id2word = vocab = opts.vocab
    _word2id = {}
    for i, w in enumerate(_id2word):
        _word2id[w] = i

    # batch word input
    # [batch_size, ]
    word_inputs = Input(batch_shape=(opts.batch_size, 1), dtype='int32')
    # batch context input
    # [batch_size, context_size]
    context_inputs = Input(batch_shape=(opts.batch_size, opts.context_size))

    final_embedding = build_final_embedding(opts, word_inputs, context_inputs)

    # TODO: Loop in batch and epoch.
    # TODO: Note: a new epoch start only when the data gatherer returns a new epoch number
    # TODO: maybe i need to use tensorflow optimizer to use nce loss

    # TODO: a evaluation model use the same layer

    # TODO: i need to build another model and input a vocab to get all the embeddings
    # TODO: see https://keras.io/getting-started/faq/#how-can-i-obtain-the-output-of-an-intermediate-layer


def read_imdb_10class_task_data(opts):
    # for fine-grained IMDB review task
    MAX_SENT_LENGTH = opts.MAX_SENT_LENGTH
    MAX_SENTS = opts.MAX_SENTS
    MAX_NUM_WORDS = opts.MAX_NUM_WORDS # Maximum number of words to work with
                          # (if set, tokenization will be restricted to the top
                          # num_words most common words in the dataset).
    VALIDATION_SPLIT = opts.VALIDATION_SPLIT

    def clean_str(string):
        """
        Tokenization/string cleaning for dataset
        Every dataset is lower cased except
        """
        string = re.sub(r"\\", "", string)
        string = re.sub(r"\'", "", string)
        string = re.sub(r"\"", "", string)
        return string.strip().lower()

    data_train = json.load(open('/home/yuanzhike/IMDB10/data.json'))

    print('start reading {0}'.format('/home/yuanzhike/IMDB10/data.json'))
    from nltk import tokenize

    reviews = []
    labels = []
    texts = []

    for idx, review_object in enumerate(data_train):
        text = review_object['review']
        text = clean_str(text.encode('ascii', 'ignore').decode('ascii'))
        texts.append(text)
        sentences = tokenize.sent_tokenize(text)
        reviews.append(sentences)

        labels.append(review_object['rating'])

    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(texts)

    data = np.zeros((len(texts), MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')

    for i, sentences in enumerate(reviews):
        for j, sent in enumerate(sentences):
            if j < MAX_SENTS:
                wordTokens = text_to_word_sequence(sent)
                k = 0
                for _, word in enumerate(wordTokens):
                    if k < MAX_SENT_LENGTH and tokenizer.word_index[word] < MAX_NUM_WORDS:
                        data[i, j, k] = tokenizer.word_index[word]
                        k = k + 1

    word_index = tokenizer.word_index
    print('Total %s unique tokens.' % len(word_index))

    labels = to_categorical(np.asarray(labels))
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

    x_train = data[:-nb_validation_samples]
    y_train = labels[:-nb_validation_samples]
    x_val = data[-nb_validation_samples:]
    y_val = labels[-nb_validation_samples:]

    print('Number of reviews for training and test')
    print(y_train.sum(axis=0))
    print(y_val.sum(axis=0))

    return word_index, x_train, y_train, x_val, y_val


def text_classification_task():
    opts = Options()

    read_text_classification_task_data = read_imdb_10class_task_data

    # Read data
    if opts.pretrained_emb is None:
        opts.pretrained_emb = "/home/yuanzhike/data/glove/glove.6B.100d.txt"

    (word_index, x_train, y_train, x_val, y_val) = \
        read_text_classification_task_data(opts)

    opts.vocab_size = len(word_index) + 1

    sentence_input = Input(shape=(opts.MAX_SENT_LENGTH,), dtype='int32')
    review_input = Input(shape=(opts.MAX_SENTS, opts.MAX_SENT_LENGTH), dtype='int32')

    embedded_sequences = build_final_embedding(opts, sentence_input)

    l_gru_w = Bidirectional(GRU(100, return_sequences=True))(embedded_sequences)
    l_dense_w = TimeDistributed(Dense(200))(l_gru_w)
    l_att_w = AttentionWithContext()(l_dense_w)
    sentEncoder = Model(sentence_input, l_att_w)

    review_input = Input(shape=(MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
    review_encoder = TimeDistributed(sentEncoder)(review_input)
    l_gru_sent = Bidirectional(GRU(100, return_sequences=True))(review_encoder)
    l_dense_sent = TimeDistributed(Dense(200))(l_gru_sent)
    l_att_sent = AttentionWithContext()(l_dense_sent)
    preds = Dense(11, activation='softmax')(l_att_sent)
    model = Model(review_input, preds)

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])

    print("model fitting - Hierachical attention network")
    model.fit(x_train, y_train, validation_data=(x_val, y_val),
              nb_epoch=10, batch_size=50)


def test_final_embedding_unit():
    opts = Options()
    # For unittest
    opts.vocab = range(25)
    opts.vocab_size = 25
    input_words = np.random.randint(25, size=(opts.batch_size,))
    input_contexts = np.random.randint(25, size=(opts.batch_size, opts.context_size))

    # batch word input
    # [batch_size, ]
    word_inputs = Input(batch_shape=(opts.batch_size, 1), dtype='int32')
    # batch context input
    # [batch_size, context_size]
    context_inputs = Input(batch_shape=(opts.batch_size, opts.context_size))

    final_embedding = build_final_embedding(opts, word_inputs, context_inputs)

    model = Model(inputs=[word_inputs, context_inputs],
                  outputs=[final_embedding])
    model.compile('rmsprop', 'mse')
    output_array = model.predict([input_words, input_contexts])
    print(output_array.shape)


def test_final_embedding_from_sentence():
    opts = Options()
    # For unittest
    opts.vocab = range(25)
    opts.vocab_size = 25
    input_contexts = np.random.randint(25, size=(32, opts.MAX_SENT_LENGTH))

    # batch context input
    # [batch_size, context_size]
    context_inputs = Input(batch_shape=(32, opts.MAX_SENT_LENGTH))

    final_embedding = build_final_embedding_from_sentence(opts, context_inputs)

    model = Model(inputs=[context_inputs],
                  outputs=[final_embedding])
    model.compile('rmsprop', 'mse')
    output_array = model.predict([input_contexts])
    print(output_array.shape)


def test_read_data():
    opts = Options()
    read_imdb_10class_task_data(opts)


if __name__ == "__main__":
    # main()
    # test_final_embedding_unit()
    # test_read_data()
    # test_final_embedding_from_sentence()
    text_classification_task()