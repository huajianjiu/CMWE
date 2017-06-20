from keras.layers.merge import _Merge
from keras import backend as K
from keras.backend.tensorflow_backend import ndim, expand_dims
import tensorflow as tf
from keras.layers import TimeDistributed


class EmbeddingDot(_Merge):
    def __init__(self, axes, sen_len, proto, normalize=False, **kwargs):
        super(EmbeddingDot, self).__init__(**kwargs)
        if not isinstance(axes, int):
            if not isinstance(axes, (list, tuple)):
                raise TypeError('Invalid type for `axes` - '
                                'should be a list or an int.')
            if len(axes) != 2:
                raise ValueError('Invalid format for `axes` - '
                                 'should contain two elements.')
            if not isinstance(axes[0], int) or not isinstance(axes[1], int):
                raise ValueError('Invalid format for `axes` - '
                                 'list elements should be "int".')
        self.axes = axes
        self.sen_len = sen_len
        self.proto = proto
        self.normalize = normalize
        self.supports_masking = True

    def build(self, input_shape):
        # Used purely for shape validation.
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError('A `Dot` layer should be called '
                             'on a list of 2 inputs.')
        shape1 = input_shape[0]
        shape2 = input_shape[1]
        if shape1 is None or shape2 is None:
            return
        if isinstance(self.axes, int):
            if self.axes < 0:
                axes = [self.axes % len(shape1), self.axes % len(shape2)]
            else:
                axes = [self.axes] * 2
        else:
            axes = self.axes
        if shape1[axes[0]] != shape2[axes[1]]:
            raise ValueError(
                'Dimension incompatibility '
                '%s != %s. ' % (shape1[axes[0]], shape2[axes[1]]) +
                'Layer shapes: %s, %s' % (shape1, shape2))

    def call(self, inputs):
        x1 = inputs[0]
        x2 = inputs[1]
        if isinstance(self.axes, int):
            if self.axes < 0:
                axes = [self.axes % K.ndim(x1), self.axes % K.ndim(x2)]
            else:
                axes = [self.axes] * 2
        else:
            axes = []
            for i in range(len(self.axes)):
                if self.axes[i] < 0:
                    axes.append(self.axes[i] % K.ndim(inputs[i]))
                else:
                    axes.append(self.axes[i])
        if self.normalize:
            x1 = K.l2_normalize(x1, axis=axes[0])
            x2 = K.l2_normalize(x2, axis=axes[1])
        # print(x1)
        # repeat l_att for broadcasting because tf cannot automatically do it for this case
        x1_tile = K.tile(x1, [1, self.sen_len])
        x1 = K.reshape(x1_tile, [tf.shape(x1)[0], self.sen_len, self.proto])
        axes[0] += 1
        output = K.batch_dot(x1, x2, axes)
        return output

    def compute_output_shape(self, input_shape):
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError('A `Dot` layer should be called '
                             'on a list of 2 inputs.')
        shape1 = list(input_shape[0])
        shape2 = list(input_shape[1])
        if isinstance(self.axes, int):
            if self.axes < 0:
                axes = [self.axes % len(shape1), self.axes % len(shape2)]
            else:
                axes = [self.axes] * 2
        else:
            axes = self.axes
        shape1.pop(axes[0])
        shape2.pop(axes[1])
        shape2.pop(0)
        output_shape = shape1 + shape2
        if len(output_shape) == 1:
            output_shape += [1]
        return tuple(output_shape)

    def compute_mask(self, inputs, mask=None):
        return None

    def get_config(self):
        config = {
            'axes': self.axes,
            'normalize': self.normalize,
        }
        base_config = super(EmbeddingDot, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def batch_dot_sen(x, y, axes=None):
    """Batchwise dot product.
    ```python
        >>> x_batch = K.ones(shape=(32, 20))
        >>> y_batch = K.ones(shape=(32, 30, 20, 50))
        >>> xy_batch_dot = K.batch_dot(x_batch, y_batch, axes=[1, 2])
        >>> K.int_shape(xy_batch_dot)
        (32, 30, 50)
    ```
    """
    if isinstance(axes, int):
        axes = (axes, axes)
    x_ndim = ndim(x)
    y_ndim = ndim(y)
    if x_ndim > y_ndim:
        diff = x_ndim - y_ndim
        y = tf.reshape(y, tf.concat([tf.shape(y), [1] * (diff)], axis=0))
    elif y_ndim > x_ndim:
        diff = y_ndim - x_ndim
        x = tf.reshape(x, tf.concat([tf.shape(x), [1] * (diff)], axis=0))
    else:
        diff = 0
    if ndim(x) == 2 and ndim(y) == 2:
        if axes[0] == axes[1]:
            out = tf.reduce_sum(tf.multiply(x, y), axes[0])
        else:
            out = tf.reduce_sum(tf.multiply(tf.transpose(x, [1, 0]), y), axes[1])
    else:
        if axes is not None:
            adj_x = None if axes[0] == ndim(x) - 1 else True
            adj_y = True if axes[1] == ndim(y) - 1 else None
        else:
            adj_x = None
            adj_y = None
        print(adj_x, adj_y)
        out = tf.matmul(x, y, adjoint_a=adj_x, adjoint_b=adj_y)
    if diff:
        if x_ndim > y_ndim:
            idx = x_ndim + y_ndim - 3
        else:
            idx = x_ndim - 1
        out = tf.squeeze(out, list(range(idx, idx + diff)))
    if ndim(out) == 1:
        out = expand_dims(out, 1)
    return out

if __name__ == "__main__":
    import numpy as np
    x_batch = tf.constant(np.ones(shape=(32, 20)))
    # issue: broadcast .20 (x[1]) to each 30(y[1])
    # reshape following tile
    x_batch = tf.tile(x_batch, [1, 30])
    x_batch = tf.reshape(x_batch, [32, 30, 20])
    y_batch = tf.constant(np.ones(shape=(32, 30, 20, 50)))
    xy_batch_dot = K.batch_dot(x_batch, y_batch, axes=[1, 2])
    K.int_shape(xy_batch_dot)
    print(tf.Session().run(xy_batch_dot).shape)  # need to be 32, 30, 50