from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import threading
import time

from six.moves import xrange  # pylint: disable=redefined-builtin

import numpy as np
import tensorflow as tf

from embedding.word2vec_optimized import word2vec, Word2Vec
from embedding.word2vec_optimized import Options as Options_

flags = tf.app.flags

flags.DEFINE_string("save_path", None, "Directory to write the model.")
flags.DEFINE_string(
    "train_data", None,
    "Training data. E.g., unzipped file http://mattmahoney.net/dc/text8.zip.")
flags.DEFINE_string(
    "eval_data", None, "Analogy questions. "
    "See README.md for how to get 'questions-words.txt'.")
flags.DEFINE_integer("embedding_size", 200, "The embedding dimension size.")
flags.DEFINE_integer(
    "epochs_to_train", 15,
    "Number of epochs to train. Each epoch processes the training data once "
    "completely.")
flags.DEFINE_float("learning_rate", 0.025, "Initial learning rate.")
flags.DEFINE_integer("num_neg_samples", 25,
                     "Negative samples per training example.")
flags.DEFINE_integer("batch_size", 500,
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
flags.DEFINE_float("subsample", 1e-3,
                   "Subsample threshold for word occurrence. Words that appear "
                   "with higher frequency will be randomly down-sampled. Set "
                   "to 0 to disable.")
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


class CMWE(Word2Vec):
    def __init__(self, options, session):
        super().__init__(options, session)

    def build_graph(self):
        """Build the model graph."""
        opts = self._options

        # The training data. A text file.
        (words, counts, words_per_epoch, current_epoch, total_words_processed,
         examples, labels) = word2vec.skipgram_word2vec(filename=opts.train_data,
                                                        batch_size=opts.batch_size,
                                                        window_size=opts.window_size,
                                                        min_count=opts.min_count,
                                                        subsample=opts.subsample)
        (opts.vocab_words, opts.vocab_counts,
         opts.words_per_epoch) = self._session.run([words, counts, words_per_epoch])
        opts.vocab_size = len(opts.vocab_words)
        print("Data file: ", opts.train_data)
        print("Vocab size: ", opts.vocab_size - 1, " + UNK")
        print("Words per epoch: ", opts.words_per_epoch)

        self._id2word = opts.vocab_words
        for i, w in enumerate(self._id2word):
            self._word2id[w] = i

