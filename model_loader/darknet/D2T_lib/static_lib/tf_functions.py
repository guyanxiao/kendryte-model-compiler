import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
from struct import unpack
from tensorflow.python.framework import graph_util

# supported data types
_data_types_ = {
    'float32': ('f', 4, tf.float32),
    'float64': ('d', 8, tf.float64),
    'float': ('f', 4, tf.float32),
    'double': ('d', 8, tf.float64),
    'int8': ('b', 1, tf.int8),
    'int16': ('i', 2, tf.int16),
    'int32': ('l', 4, tf.int32),
    'uint8': ('B', 1, tf.uint8),
    'uint16': ('I', 2, tf.uint16),
    'uint32': ('L', 4, tf.uint32)
}


# tf nn functions
def leaky_relu(inputs, alpha=0.1, name='leaky_relu'):
    return tf.nn.leaky_relu(inputs, alpha=alpha, name=name)


def Batch_Normalize(data, scale, mean, variance, epsilon=1e-6,
                    scope="Batch_Normalize",
                    reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        return tf.multiply((data - mean) / (tf.sqrt(variance) + epsilon), scale)


# packed tf layer functions
def convolutional(net, weights, biases,
                  strides=1,
                  padding='SAME',
                  activation_fn=None,
                  batch_normalize=False,
                  bn_scale=None,
                  bn_mean=None,
                  bn_variance=None,
                  parent_scope=None, my_scope="conv2d",
                  reuse=False):
    with tf.variable_scope(None if parent_scope == 'None' else parent_scope,
                           my_scope, reuse=reuse):
        my_out = tf.nn.conv2d(net, weights,
                              strides=[1, strides, strides, 1],
                              padding=padding,
                              name="conv2d")

        if batch_normalize:
            my_out = Batch_Normalize(my_out, bn_scale, bn_mean, bn_variance,
                                     scope="BN", reuse=reuse)

        my_out = tf.add(my_out, biases, name="bias")
        if activation_fn is not None:
            my_out = activation_fn(my_out)

        return my_out


def depthwise_convolutional(net, weights, biases,
                            strides=1,
                            padding='SAME',
                            activation_fn=None,
                            batch_normalize=False,
                            bn_scale=None,
                            bn_mean=None,
                            bn_variance=None,
                            parent_scope=None, my_scope="dw_conv2d",
                            reuse=False):
    with tf.variable_scope(None if parent_scope == 'None' else parent_scope,
                           my_scope, reuse=reuse):
        my_out = tf.nn.depthwise_conv2d(net, weights,
                                        strides=[1, strides, strides, 1],
                                        padding=padding,
                                        name="dw_conv2d")

        if batch_normalize:
            my_out = Batch_Normalize(my_out, bn_scale, bn_mean, bn_variance,
                                     scope="BN", reuse=reuse)

        my_out = tf.add(my_out, biases, name="bias")
        if activation_fn is not None:
            my_out = activation_fn(my_out)

        return my_out


def max_pool(net, ksize=2, strides=1, padding='SAME', scope=None, name="max_pool"):
    with tf.variable_scope(name if scope == 'None' else scope):
        return tf.nn.max_pool(net, ksize=[1, ksize, ksize, 1],
                              strides=[1, strides, strides, 1],
                              padding=padding, name=name)


def avg_pool(net, ksize=2, strides=1, padding='SAME', scope=None, name="avg_pool"):
    with tf.variable_scope(name if scope == 'None' else scope):
        return tf.nn.avg_pool(net, ksize=[1, ksize, ksize, 1],
                              strides=[1, strides, strides, 1],
                              padding=padding, name=name)


def route_concat(layers_route, axis=-1, name='Route_concat'):
    len_dim = len(layers_route[0].shape)
    concat_dim = (len_dim + axis) % len_dim
    return tf.concat(layers_route, concat_dim, name=name)


def route_sum(layers_route, activation_fn, scope='Route_sum'):
    with tf.variable_scope(scope):
        return activation_fn(tf.add_n(layers_route, name='sum'))


# I/O
def bytes_from_TFW(tfw_file):
    with open(tfw_file, 'rb') as F:
        return F.read(-1)


def var_from_bytes(tfw_bytes, start, end,
                   resize_as=None,
                   trainable=False,
                   name=None,
                   dtype=tf.float32):
    assert dtype in _data_types_, 'check the supported data types: ' + \
                                  ''.join('\n<%s>' % s for s in _data_types_.keys())
    count = (end - start) / _data_types_[dtype][1]
    arr = unpack('%i%s' % (count, _data_types_[dtype][0]), tfw_bytes[start:end])
    if resize_as:
        arr = np.resize(arr, resize_as)

    return tf.Variable(arr, trainable=trainable, dtype=dtype, name=name)


def freeze_to_PB(session, out_nodes_list, out_path):
    LAST_DOT = out_path.find('/')
    out_dir = './' if LAST_DOT < 0 else out_path[:LAST_DOT + 1]
    out_name = out_path[LAST_DOT + 1:]
    constant_graph = graph_util.convert_variables_to_constants(session,
                                                               session.graph_def,
                                                               output_node_names=out_nodes_list)
    tf.train.write_graph(constant_graph, out_dir, out_name, as_text=False)
