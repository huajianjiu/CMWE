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

from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializations

from embedding.word2vec import word2vec, Word2Vec
from embedding.word2vec import Options as Options_
from mulembedding import MulEmbedding
from attention import AttLayer

flags = tf.app.flags

flags.DEFINE_string("save_path", None, "Directory to write the model.")
flags.DEFINE_string(
    "train_data", None,
    "Training data. E.g., unzipped file http://mattmahoney.net/dc/text8.zip.")
flags.DEFINE_string(
    "eval_data", None, "Analogy questions. "
                       "See README.md for how to get 'questions-words.txt'.")
flags.DEFINE_string(
    "pretrained_emb", "embedding/glove.6B.100d.txt",
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


if __name__ == "__main__":
    with tf.Graph().as_default(), tf.Session() as session:
        with tf.device("/cpu:0"):
            (words, counts, words_per_epoch, _epoch, _words, examples,
             labels, contexts) = word2vec.context_skipgram_word2vec(filename="test.corpus",
                                                                    batch_size=5,
                                                                    window_size=3,
                                                                    min_count=1,
                                                                    subsample=1e-4)
            (vocab, a, b, c) = session.run([words, examples, labels, contexts])
