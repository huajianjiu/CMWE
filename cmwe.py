"""
by Yuanzhi Ke. June 2017
require keras 1.2.2, tensorflow 1.1, numpy
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import os

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Dense, Input
from keras.layers import Embedding, Merge, Dropout, GRU, Bidirectional, TimeDistributed
from keras.layers.merge import Dot
from keras.models import Model

from mulembedding import MulEmbedding
from attention import AttentionWithContext

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
flags.DEFINE_integer("prototypes", 20,
                     "The number of prototypes of each word.")
flags.DEFINE_float("subsample", 1e-4,
                   "Subsample threshold for word occurrence. Words that appear "
                   "with higher frequency will be randomly down-sampled. Set "
                   "to 0 to disable.")
flags.DEFINE_integer("context_size", 10,
                     "The context, size. If you change it, you"
                     "Need to also change it in embeddings/word2vec_kernels.cc now.")
flags.DEFINE_integer("rnn_time_steps", 10,
                     "The time steps for the rnn.")


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
        self.time_steps = FLAGS.rnn_time_steps

# TODO: As i do not know how to make the keras auto use the tensorflow data gatherer
# TODO: so I need to do it in keras way
# TODO: fix the rank 3 2 error (maybe caused by the problem of the tensorflow gatherer

def main():
    word2vec = tf.load_op_library(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'embedding/word2vec_ops.so'))
    opts = Options()
    with tf.Graph().as_default(), tf.Session() as session:
        with tf.device("/cpu:0"):
            (words, counts, words_per_epoch, _epoch, _words, examples,
             labels, contexts) = word2vec.full_context_skipgram_word2vec(filename=opts.train_data,
                                                                         batch_size=opts.batch_size,
                                                                         window_size=opts.window_size,
                                                                         min_count=opts.min_count,
                                                                         subsample=opts.subsample)
            (vocab, input_words, target_labels, input_contexts) = \
                session.run([words, examples, labels, contexts])

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
    multiple_embedding_layer = MulEmbedding(input_dim=len(vocab) + 1,
                                            output_prototypes=opts.prototypes,
                                            output_dim=opts.emb_dim,
                                            input_length=opts.batch_size,
                                            trainable=True)

    # batch word input
    # [batch_size, ]
    batch_word_input = Input(shape=(opts.batch_size,), dtype='int32')
    # [batch_size, prototypes, emb_dim]
    multiple_embeded_batch = multiple_embedding_layer(batch_word_input)

    # batch context input
    # [batch_size, context_size]
    batch_context_input = Input(shape=(opts.batch_size, opts.context_size))
    embedded_batch_context = single_embedding_layer(batch_context_input)
    l_lstm = Bidirectional(GRU(opts.time_steps, return_sequences=True))(embedded_batch_context)
    # [batch_size, prototypes]
    l_dense = TimeDistributed(Dense(opts.prototypes))(l_lstm)
    l_att = AttentionWithContext()(l_dense)

    # get final embedding
    embedding_dot = Dot(axes=[1,1])
    final_embedding = Dot([multiple_embeded_batch, l_att])

    model = Model(inputs=[batch_word_input, batch_context_input],
                  outputs=[final_embedding])
    model.compile('rmsprop', 'mse')
    output_array = model.predict([input_words, input_contexts])
    print(output_array[0].shape)

if __name__ == "__main__":
    main()

