# author - Richard Liao
# Dec 26 2016
# Attention GRU network

from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers

# TODO: change to fit the cnn-based model
class AttLayer(Layer):
    def __init__(self, **kwargs):
        self.init = initializers.get('normal')
        # self.input_spec = [InputSpec(ndim=3)]
        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        # self.W = self.init((input_shape[-1],1))
        self.W = self.init((input_shape[-1],))
        # self.input_spec = [InputSpec(shape=input_shape)]
        self.trainable_weights = [self.W]
        super(AttLayer, self).build(input_shape)  # be sure you call this somewhere!

    def call(self, x, mask=None):
        eij = K.tanh(K.dot(x, self.W))

        ai = K.exp(eij)
        weights = ai / K.sum(ai, axis=1).dimshuffle(0, 'x')

        weighted_input = x * weights.dimshuffle(0, 1, 'x')
        return weighted_input.sum(axis=1)

    def get_output_shape_for(self, input_shape):
        return input_shape[0], input_shape[-1]

if __name__ == "__main__":
    from keras.models import Sequential
    import numpy as np

    model = Sequential()
    model.add(AttLayer(input_shape=[2,3]))
    input_array = np.random.randint(3, size=(32, 10))
    model.compile('rmsprop', 'mse')
    output_array = model.predict(input_array)
    assert output_array.shape == (32, 10, 4, 5)