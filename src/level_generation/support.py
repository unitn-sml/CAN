import tensorflow as tf

def _soft_argmax(x, sample_channels, beta=1e4):
    """
    Find the argmax in a derivable way thanks to the use of an extremely stretched softmax
    """
    x_range = tf.range(sample_channels, dtype=x.dtype)
    return tf.reduce_sum(tf.multiply(tf.nn.softmax(x * beta, axis=-1), x_range), axis=-1)

def _soft_onehotcategorical(x, beta=1e4):
    """
    Stretch the softmax with beta to obtain a onehot-categorical like output but derivable
    """
    return tf.nn.softmax(x * beta)
