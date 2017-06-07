from keras import backend as K
from keras.engine.topology import Layer

class Embedding3D(Layer):
    """
    For Multiprototype embedding, 
    Turns positive integers (indexes) into dense vectors of fixed size.
    This layer can only be used as the first layer in a model.
    # Arguments
      input_dim: int > 0. Size of the vocabulary,
          i.e. maximum integer index + 1.
      output_dim1: int >= 0. 1st dimension of the dense embedding.
      output_dim2: int >= 0. 2nd dimension of the dense embedding.
      embeddings_initializer: Initializer for the `embeddings` tensor
          (see [initializers](../initializers.md)).
      embeddings_regularizer: Regularizer function applied to
          the `embeddings` tensor
          (see [regularizer](../regularizers.md)).
      embeddings_constraint: Constraint function applied to
          the `embeddings` matrix
          (see [constraints](../constraints.md)).
      mask_zero: Whether or not the input value 0 is a special "padding"
          value that should be masked out.
          This is useful when using [recurrent layers](recurrent.md)
          which may take variable length input.
          If this is `True` then all subsequent layers
          in the model need to support masking or an exception will be raised.
          If mask_zero is set to True, as a consequence, index 0 cannot be
          used in the vocabulary (input_dim should equal size of
          vocabulary + 1).
      input_length: Length of input sequences, when it is constant.
          This argument is required if you are going to connect
          `Flatten` then `Dense` layers upstream
          (without it, the shape of the dense outputs cannot be computed).
    # Input shape
        2D tensor with shape: `(batch_size, sequence_length)`.
    # Output shape
        3D tensor with shape: `(batch_size, sequence_length, output_dim)`.
    # References
        - [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)
    """