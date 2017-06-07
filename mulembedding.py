from keras import backend as K
from keras import initializations, regularizers, constraints
from keras.engine import Layer

# TODO: return multiprototype embedding
class MulEmbedding(Layer):
    input_ndim = 2

    def __init__(self, input_dim, output_dim,
                 init='uniform', input_length=None,
                 W_regularizer=None, activity_regularizer=None,
                 W_constraint=None,
                 mask_zero=False,
                 weights=None, dropout=0., **kwargs):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.input_length = input_length
        self.mask_zero = mask_zero
        self.dropout = dropout

        self.W_constraint = constraints.get(W_constraint)
        self.constraints = [self.W_constraint]

        self.W_regularizer = regularizers.get(W_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        if 0. < self.dropout < 1.:
            self.uses_learning_phase = True
        self.initial_weights = weights
        kwargs['input_shape'] = (self.input_length,)
        kwargs['input_dtype'] = 'int32'
        super(MulEmbedding, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.init((self.input_dim, self.output_dim),
                           name='{}_W'.format(self.name))
        self.trainable_weights = [self.W]
        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)

    def compute_mask(self, x, mask=None):
        if not self.mask_zero:
            return None
        else:
            return K.not_equal(x, 0)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.input_length, self.output_dim)

    def call(self, x, mask=None):
        if 0. < self.dropout < 1.:
            retain_p = 1. - self.dropout
            B = K.random_binomial((self.input_dim,), p=retain_p) * (1. / retain_p)
            B = K.expand_dims(B)
            W = K.in_train_phase(self.W * B, self.W)
        else:
            W = self.W
        out = K.gather(W, x)
        return out

    def get_config(self):
        config = {'input_dim': self.input_dim,
                  'output_dim': self.output_dim,
                  'init': self.init.__name__,
                  'input_length': self.input_length,
                  'mask_zero': self.mask_zero,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'dropout': self.dropout}
        base_config = super(MulEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))