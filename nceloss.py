import tensorflow as tf
from keras import backend as K


def nce_loss(true_logits, sampled_logits, opts):
    """Build the graph for the NCE loss."""

    # cross-entropy(logits, labels)
    true_xent = K.categorical_crossentropy(
        target=tf.ones_like(true_logits), output=true_logits)
    sampled_xent = K.categorical_crossentropy(
        target=tf.zeros_like(sampled_logits), output=sampled_logits)

    # NCE-loss is the sum of the true and noise (sampled words)
    # contributions, averaged over the batch.
    nce_loss_tensor = (K.sum(true_xent) +
                       K.sum(sampled_xent)) / opts.batch_size
    return nce_loss_tensor
