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

from embedding.word2vec import word2vec, Word2Vec
from embedding.word2vec import Options as Options_

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
tf.flags.DEFINE_integer("filter_sizes", 5, "filter size(default: 5")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
flags.DEFINE_boolean(
    "interactive", False,
    "If true, enters an IPython interactive session to play with the trained "
    "model. E.g., try model.analogy(b'france', b'paris', b'russia') and "
    "model.nearby([b'proton', b'elephant', b'maxwell'])")

FLAGS = flags.FLAGS

CONTEXT_WINDOW = 5


class Options(Options_):
    def __init__(self):
        super().__init__()
        self.prototypes = FLAGS.prototypes
        self.pretrained_emb = FLAGS.pretrained_emb
        self.filter_size = FLAGS.filter_size
        self.context_size = CONTEXT_WINDOW * 2 + 1


class CMWE(Word2Vec):
    def __init__(self, options, session):
        super().__init__(options, session)
        self.init_single_emb(session)

    def init_single_emb(self, session):
        opts = self._options
        single_embedding_placeholder = tf.placeholder(tf.float32, [opts.vocab_size, opts.emb_dim])
        embedding_init = self._single_emb.assign(single_embedding_placeholder)
        single_embedding = np.zeros([opts.vocab_size, opts.emb_dim],
                                    dtype="float32")
        with open(opts.pretrained_emb, 'r') as f:
            for line in f.readlines():
                # here the build graph is inited so self has id2word and word2id
                if len(line.split()) < 3:
                    continue
                line = line.split()
                if line[0].strip() in opts.vocab_words:
                    single_embedding[self._word2id[line[0].strip()]] = np.array(line[1:-1], dtype="float32")
        session.run(embedding_init,
                    feed_dict={single_embedding_placeholder: single_embedding})

    def forward(self, examples, labels):
        """Build the graph for the forward pass."""
        opts = self._options

        # Declare all variables we need.
        # Multi-prototype Embedding: [vocab_size, prototypes, emb_dim]
        init_width = 0.5 / opts.emb_dim
        mul_emb = tf.Variable(
            tf.random_uniform(
                [opts.vocab_size, opts.prototypes, opts.emb_dim], -init_width, init_width),
            name="mul_emb")
        self._mul_emb = mul_emb

        # Pretrained single prototype embedding
        single_emb = \
            tf.Variable(tf.constant(0.0, shape=[opts.vocab_size,
                                                opts.emb_dim]),
                        trainable=False, name="single_emb")
        self._single_emb = single_emb

        # TODO: change to fit the context_size. and all use one set
        # cnn paramters
        filter_shape = [opts.vocab_size, opts.filter_size, opts.emb_dim, 1, opts.num_filters]
        conv1_w = tf.Variable(
            tf.truncated_normal(filter_shape, stddev=0.1),
            name="conv1_w"
        )
        conv1_b = tf.Variable(tf.constant(0.1, shape=[opts.vocab_size, opts.num_filters]),
                              name="conv1_b")
        conv2_w = tf.Variable(
            tf.truncated_normal(filter_shape, stddev=0.1),
            name="conv2_w"
        )
        conv2_b = tf.Variable(tf.constant(0.1, shape=[opts.vocab_size, opts.num_filters]),
                              name="conv2_b")
        conv3_w = tf.Variable(
            tf.truncated_normal(filter_shape, stddev=0.1),
            name="conv3_w"
        )
        conv3_b = tf.Variable(tf.constant(0.1, shape=[opts.vocab_size, opts.num_filters]),
                              name="conv3_b")
        dense_w = tf.Variable(
            tf.random_uniform([opts.num_filters, opts.prototypes]),
            name="dense_w"
        )
        dense_b = tf.Variable(tf.constant(0.1, shape=[opts.prototypes]), name="dense_b")

        # Softmax weight: [vocab_size, emb_dim]. Transposed.
        sm_w_t = tf.Variable(
            tf.zeros([opts.vocab_size, opts.emb_dim]),
            name="sm_w_t")

        # Softmax bias: [vocab_size].
        sm_b = tf.Variable(tf.zeros([opts.vocab_size]), name="sm_b")

        # Global step: scalar, i.e., shape [].
        self.global_step = tf.Variable(0, name="global_step")

        # Nodes to compute the nce loss w/ candidate sampling.
        labels_matrix = tf.reshape(
            tf.cast(labels,
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

        # Embeddings for examples: [batch_size, emb_dim]
        example_mulemb = tf.nn.embedding_lookup(mul_emb, examples)

        # CNN paramter for labels
        example_conv1_w = tf.nn.embedding_lookup(conv1_w, examples)
        example_conv2_w = tf.nn.embedding_lookup(conv2_w, examples)
        example_conv3_w = tf.nn.embedding_lookup(conv3_w, examples)
        example_conv1_b = tf.nn.embedding_lookup(conv1_b, examples)
        example_conv2_b = tf.nn.embedding_lookup(conv1_b, examples)
        example_conv3_b = tf.nn.embedding_lookup(conv1_b, examples)

        # Weights for labels: [batch_size, emb_dim]
        true_w = tf.nn.embedding_lookup(sm_w_t, labels)
        # Biases for labels: [batch_size, 1]
        true_b = tf.nn.embedding_lookup(sm_b, labels)

        # Weights for sampled ids: [num_sampled, emb_dim]
        sampled_w = tf.nn.embedding_lookup(sm_w_t, sampled_ids)
        # Biases for sampled ids: [num_sampled, 1]
        sampled_b = tf.nn.embedding_lookup(sm_b, sampled_ids)

        # True logits: [batch_size, 1]
        true_logits = tf.reduce_sum(tf.multiply(example_emb, true_w), 1) + true_b

        # Sampled logits: [batch_size, num_sampled]
        # We replicate sampled noise labels for all examples in the batch
        # using the matmul.
        sampled_b_vec = tf.reshape(sampled_b, [opts.num_samples])
        sampled_logits = tf.matmul(example_emb,
                                   sampled_w,
                                   transpose_b=True) + sampled_b_vec
        return true_logits, sampled_logits

    def build_graph(self):
        """Build the graph for the full model."""
        opts = self._options
        # The training data. A text file.
        (words, counts, words_per_epoch, self._epoch, self._words, examples,
         labels, contexts) = word2vec.full_context_skipgram_word2vec(filename=opts.train_data,
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
        self._examples = examples
        self._labels = labels
        self._contexts = contexts
        self._id2word = opts.vocab_words
        for i, w in enumerate(self._id2word):
            self._word2id[w] = i
        true_logits, sampled_logits = self.forward(examples, labels)
        loss = self.nce_loss(true_logits, sampled_logits)
        tf.summary.scalar("NCE loss", loss)
        self._loss = loss
        self.optimize(loss)

        # Properly initialize all variables.
        tf.global_variables_initializer().run()

        self.saver = tf.train.Saver()
