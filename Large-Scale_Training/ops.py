import tensorflow as tf

'''Operations'''


def conv2d(x, kernel, bias, strides=1, scope=None, activation=None):
    with tf.variable_scope(scope):
        out = tf.nn.conv2d(x,kernel,[1,strides,strides,1],padding='SAME', name='conv2d')
        out = tf.nn.bias_add(out,bias, name='BiasAdd')

        if activation is None:
            return out
        elif activation is 'ReLU':
            return tf.nn.relu(out)
        elif activation is 'leakyReLU':
            return tf.nn.leaky_relu(out, 0.2)

def dense(x, weights, bias, scope=None, activation=None, reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        out=tf.matmul(x, weights,name='dense')
        out=tf.nn.bias_add(out,bias,name='BiasAdd')

        if activation is None:
            return out
        elif activation is 'ReLU':
            return tf.nn.relu(out)
        elif activation is 'leakyReLU':
            return tf.nn.leaky_relu(out, 0.2)