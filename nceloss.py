import tensorflow as tf
from keras import backend as K


def nce_loss(true_logits, sampled_logits, opts, from_logits=True):
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
