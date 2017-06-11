import tensorflow as tf
from keras import backend as K


# TODO: change to fit the format of keras losses
# TODO: maybe we should generate and label samples in the model and compute the sigmoid loss
def _nce_loss(true_logits, sampled_logits, opts, from_logits=True):
    """Build the graph for the NCE loss."""
    # if from_logits is True (default). the logits is expected to be logits tensors. otherwise, probability distribution
    # Note: the keras defualt of binary_crossentropy is probablity distribution.
    # cross-entropy(logits, labels)
    true_xent = K.binary_crossentropy(
        target=tf.ones_like(true_logits), output=true_logits, from_logits=from_logits)
    sampled_xent = K.binary_crossentropy(
        target=tf.zeros_like(sampled_logits), output=sampled_logits, from_logits=from_logits)

    # NCE-loss is the sum of the true and noise (sampled words)
    # contributions, averaged over the batch.
    nce_loss_tensor = (K.sum(true_xent) +
                       K.sum(sampled_xent)) / opts.batch_size
    return nce_loss_tensor


def nce_loss(y_true, y_pred):
    pass


def generate_negative_samples(labels_matrix, num_samples, vocab_size, vocab_counts):
    sampled_ids, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
        true_classes=labels_matrix,
        num_true=1,
        num_sampled=num_samples,
        unique=True,
        range_max=vocab_size,
        distortion=0.75,
        unigrams=vocab_counts.tolist()))
    return sampled_ids
