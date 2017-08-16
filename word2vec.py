# keras word2vec
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

from keras.layers import Embedding, Input
from embedding.word2vec import Options

word2vec = tf.load_op_library(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'word2vec_ops.so'))

flags = tf.app.flags

flags.DEFINE_string("save_path", None, "Directory to write the model.")
flags.DEFINE_string(
    "train_data", None,
    "Training data. E.g., unzipped file http://mattmahoney.net/dc/text8.zip.")
flags.DEFINE_string(
    "eval_data", None, "Analogy questions. "
    "See README.md for how to get 'questions-words.txt'.")
flags.DEFINE_integer("embedding_size", 100, "The embedding dimension size.")
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

if not FLAGS.train_data or not FLAGS.eval_data or not FLAGS.save_path:
    print("--train_data --eval_data and --save_path must be specified.")
    sys.exit(1)
opts = Options()

with tf.Graph().as_default(), tf.Session() as session:
    with tf.device("/cpu:0"):
        (words, counts, words_per_epoch, _epoch, _words, examples,
         labels, contexts) = word2vec.context_skipgram_word2vec(filename="test.corpus",
                                                                batch_size=opts.batch_size,
                                                                window_size=opts.window_size,
                                                                min_count=opts.min_count,
                                                                subsample=opts.subsample)
        (vocab, a, b, c) = session.run([words, examples, labels, contexts])

init_width = 0.5 / opts.emb_dim
word_embedding_matrix = np.random.uniform( -init_width, init_width,(len(vocab) + 1, opts.emb_dim))

embedding_layer = Embedding(len(vocab) + 1,
                            opts.emb_dim,
                            weights=[word_embedding_matrix],
                            input_length=opts.batch_size,
                            trainable=True)
batch_input = Input(shape=(opts.batch_size,), dtype='int32')
embeded = embedding_layer(batch_input)
