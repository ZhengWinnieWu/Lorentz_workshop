import tensorflow as tf    
#tf.compat.v1.disable_v2_behavior() # <-- HERE !

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import regularizers
import tensorflow.keras.backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dropout, Activation, Reshape, Flatten, LSTM, Dense, Dropout, Embedding, Bidirectional, GRU
from tensorflow.keras import Sequential
from tensorflow.keras import initializers, regularizers
from tensorflow.keras import optimizers
from tensorflow.keras import constraints
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten

def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)
    
class AttentionWithContext(Layer):
    """
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.
    follows these equations:
    
    (1) u_t = tanh(W h_t + b)
    (2) \alpha_t = \frac{exp(u^T u)}{\sum_t(exp(u_t^T u))}, this is the attention weight
    (3) v_t = \alpha_t * h_t, v in time t
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        3D tensor with shape: `(samples, steps, features)`.
    """

    def __init__(self,
                W_regularizer=None, u_regularizer=None, b_regularizer=None,
                W_constraint=None, u_constraint=None, b_constraint=None,
                bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
                'W_regularizer': self.W_regularizer,
                'u_regularizer': self.u_regularizer,
                'b_regularizer': self.b_regularizer,
                'W_constraint': self.W_constraint,
                'u_constraint': self.u_constraint,
                'b_constraint': self.b_constraint,
                'bias': self.bias,
        })
        return config

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1],),
                                initializer=self.init,
                                name='{}_W'.format(self.name),
                                regularizer=self.W_regularizer,
                                constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight(shape=(input_shape[-1],),
                                    initializer='zero',
                                    name='{}_b'.format(self.name),
                                    regularizer=self.b_regularizer,
                                    constraint=self.b_constraint)

        self.u = self.add_weight(shape=(input_shape[-1],),
                                initializer=self.init,
                                name='{}_u'.format(self.name),
                                regularizer=self.u_regularizer,
                                constraint=self.u_constraint)

        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = dot_product(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)

        a = K.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero and this results in NaN's. 
        # Should add a small epsilon as the workaround
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        
        return weighted_input, a

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], input_shape[2]
    
class Addition(Layer):
    """
    This layer is supposed to add of all activation weight.
    We split this from AttentionWithContext to help us getting the activation weights
    follows this equation:
    (1) v = \sum_t(\alpha_t * h_t)
    
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    """

    def __init__(self, **kwargs):
        super(Addition, self).__init__(**kwargs)

    def build(self, input_shape):
        self.output_dim = input_shape[-1]
        super(Addition, self).build(input_shape)

    def call(self, x):
        return K.sum(x, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
    
def build_lstm(ntimestep, nfeature, **kwargs):
    
    reg_val = kwargs.get('regval', [1, 0.01])
    numlayer = kwargs.get('layers',2)
    neurons = kwargs.get()

    input_tensor = Input(shape=(ntimestep, nfeature))
    layerlstm = layers.LSTM(100, return_sequences=True, kernel_regularizer=regularizers.l2(regval[0])(input_tensor)
    if numlayer >= 2:
        layer1 = layers.LSTM(20, return_sequences=True, kernel_regularizer=regularizers.l2(regval[1]))(layer1)
        for i in range(numlayer):
            layer1 = layers.LSTM(20, return_sequences=True, kernel_regularizer=regularizers.l2(regval[i]))(layer1)

    layer1, alfa = AttentionWithContext()(layerlstm)
    layer1 = Addition()(layer1)
    layer1 = layers.Dense(5, activation="relu")(layer1)
    output_tensor = layers.Dense(2, activation='softmax')(layer1)

    model = Model(input_tensor, output_tensor)
    model.summary()                        
    return model
                            
def build_CNN(
        inputs: tf.keras.Input ,
        output_shape: float,
        **kwargs):
    """
    Builds CNN architecture based on the defined params.
    Baseline structure 2 layer CNN.
    
    input_shape: tuple shape 
    output: float
    kwargs: list of HPs for the network
    """
                            
    random_network_seed = kwargs.get('random_network_seed', None)
    filt_size = kwargs.get('filt',[3, 3])
    insize = kwargs.get('ins',[6,32])
    reg_val = kwargs.get('regval', [0.01, 0.01])
    numlayer = kwargs.get('CNNlayers',2)

    # inputs = Input(shape= *input_shape)
    x = Conv2D(in_size[0], (filt_size[0], filt_size[0]), padding='same', stride = kwargs.get('stride',2))(inputs)
    if kwargs.get('maxPool',0):
        x = MaxPooling2D((2, 2), strides=2)(x)
    if numlayer >= 2:
        for i in range(1,numlayer):
            x = Conv2D(in_size[i], (filt_size[i], filt_size[i]), padding='same', stride = kwargs.get('stride',2))(x)
            if kwargs.get('maxPool',0):
                x = MaxPooling2D((2, 2), strides=2)(x)
                            
    if kwargs.get('dense',0):
        flat1 = layers.Flatten()(x)
        output = Dense(kwargs.get('output_shape', 5), activation='relu', use_bias=True,
                        kernel_regularizer=regularizers.l1_l2(l1=reg_val[0], l2=reg_val[0]),
                        bias_initializer=keras.initializers.RandomNormal(seed=random_network_seed),
                        kernel_initializer=keras.initializers.RandomNormal(seed=random_network_seed), name='dense_out')(flat1)
    else:
        output = GlobalAveragePooling2D()(x)
                            
    # model = Model(inputs, output)

                            
    return output
                            
def create_multi_Inp(data: Dict)
        inputs = []                  
        for keys, vals in data:
                inputs.append(Input(shape= *vals, name = keys))
        return inputs
        
                            
def assemble_network(
            data: Dict
            input_shape: tuple,
            regions: float,
            ntimestep: float, 
            **kwargs):
                            
    inputs = create_multi_Inp(data)       
    features = []                  
    for mod in range(regions)
        model, outs = build_CNN(inputs[mod], **kwargs)                  
        features.append(outs)
    x = layers.Concatenate(axis = -1)(features)
    lstm = build_lstm(ntimestep, regions, **kwargs)     
    output = lstm(x)
    
                            
    return Model(inputs, output)