"""
by Yuanzhi Ke. June 2017
require keras 1.2.2, tensorflow 1.1, numpy
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional
from keras.models import Model

from embedding.word2vec import word2vec, Word2Vec
from embedding.word2vec import Options as Options_
from mulembedding import MulEmbedding
from attention import AttentionWithContext

flags = tf.app.flags

flags.DEFINE_string("save_path", None, "Directory to write the model.")
flags.DEFINE_string(
    "train_data", None,
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
                     "Numbers of training examples each step processes "
                     "(no minibatching).")
flags.DEFINE_integer("concurrent_steps", 12,
                     "The number of concurrent training steps.")
flags.DEFINE_integer("window_size", 5,
                     "The number of words to predict to the left and right "
                     "of the target word.")
flags.DEFINE_integer("min_count", 5,
                     "The minimum number of word occurrences for it to be "
                     "included in the vocabulary.")
flags.DEFINE_integer("prototypes", 10,
                     "The number of prototypes of each word.")
flags.DEFINE_float("subsample", 1e-4,
                   "Subsample threshold for word occurrence. Words that appear "
                   "with higher frequency will be randomly down-sampled. Set "
                   "to 0 to disable.")
tf.flags.DEFINE_string("filter_sizes", "5,5,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
flags.DEFINE_boolean(
    "interactive", False,
    "If true, enters an IPython interactive session to play with the trained "
    "model. E.g., try model.analogy(b'france', b'paris', b'russia') and "
    "model.nearby([b'proton', b'elephant', b'maxwell'])")

FLAGS = flags.FLAGS


class Options(Options_):
    def __init__(self):
        super().__init__()
        self.prototypes = FLAGS.prototypes
        self.pretrained_emb = FLAGS.pretrained_emb


def main(_):
    opts = Options()
    with tf.Graph().as_default(), tf.Session() as session:
        with tf.device("/cpu:0"):
            (words, counts, words_per_epoch, _epoch, _words, examples,
             labels, contexts) = word2vec.full_context_skipgram_word2vec(filename="test.corpus",
                                                                         batch_size=opts.batch_size,
                                                                         window_size=opts.window_size,
                                                                         min_count=opts.min_count,
                                                                         subsample=opts.subsample)
            (vocab, a, b, c) = session.run([words, examples, labels, contexts])

    _id2word = vocab
    _word2id = {}
    for i, w in enumerate(_id2word):
        _word2id[w] = i

    # create single prototype embedding layer
    if opts.pretrained_emb is None:
        single_embedding_layer = Embedding(input_dim=len(vocab) + 1,
                                           output_dim=opts.emb_dim,
                                           input_length=opts.batch_size,
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

        single_embedding_layer = Embedding(len(vocab) + 1,
                                           opts.emb_dim,
                                           weights=[single_embedding_matrix],
                                           input_length=opts.batch_size,
                                           trainable=False)

    # create multiple prototype embedding layer
    multiple_embedding_layer = MulEmbedding(input_dim=len(vocab)+1,
                                            output_prototypes=opts.prototypes,
                                            output_dim=opts.emb_dim,
                                            input_length=opts.batch_size,
                                            trainable=True)

    # [batch_size, 1]
    batch_word_input = Input(shape=(opts.batch_size, 1), dtype='int32')
    # [batch_size, prototypes, emb_dim]
    multiple_embeded_batch = multiple_embedding_layer(batch_word_input)



