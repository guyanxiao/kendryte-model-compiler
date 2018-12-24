'''
 * Copyright 2018 Canaan Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 '''

import tensorflow as tf
import numpy as np


class LayerBase:
    def __init__(self):
        self.name = 'no_name'
        self.config = {}

    def type_match(self, nodes, type_list):
        if len(nodes) != len(type_list):
            return False
        else:
            for node, ty in zip(nodes, type_list):
                if node.op.type != ty:
                    return False
        return True


class LayerNet(LayerBase):
    def __init__(self, sess, info):
        super().__init__()
        self.name = 'net'
        self.config = {}
        self.tensor = info
        x, = info

        _, self.config['width'], self.config['height'], self.config['channels'] = x.shape.as_list()
        self.config['batch'] = 1
        self.config['subdivisions'] = 1


class LayerConvolutional(LayerBase):
    def __init__(self, sess, dataset, info):
        super().__init__()
        self.name = 'convolutional'
        self.config = {}
        self.tensor = info
        self.bias = None
        self.batch_normalize_epsilon = 0
        batch_norm = None
        activation = None
        bias_add = None
        bn_add, bn_sub, bn_div, bn_mul = None, None, None, None
        leaky_reul_mul = None

        if self.type_match(info, ['Add', 'Conv2D']):
            bias_add, conv2d = info
        elif self.type_match(info, ['Add', 'Mul', 'Conv2D']):
            bn_add, bn_mul, conv2d = info
            bn_div, bn_sub = 1, 0
            batch_norm = [bn_add, bn_mul, bn_div, bn_sub]
        elif self.type_match(info, ['BiasAdd', 'Conv2D']):
            bias_add, conv2d = info
        elif self.type_match(info, ['Relu', 'BiasAdd', 'Conv2D']):
            activation, bias_add, conv2d = info
        elif self.type_match(info, ['Relu', 'FusedBatchNorm', 'Conv2D']):
            activation, batch_norm, conv2d = info
        elif self.type_match(info, ['Relu', 'FusedBatchNorm', 'BiasAdd', 'Conv2D']):
            activation, batch_norm, bias_add, conv2d = info
        elif self.type_match(info, ['LeakyRelu', 'BiasAdd', 'Conv2D']):
            activation, bias_add, conv2d = info
        elif self.type_match(info, ['Maximum', 'Mul', 'BiasAdd', 'Conv2D']):
            leaky_reul_max, leaky_reul_mul, bias_add, conv2d = info
            activation = ['leaky', leaky_reul_max, leaky_reul_mul]
        elif self.type_match(info, ['Maximum', 'Mul', 'Add', 'Conv2D']):
            leaky_reul_max, leaky_reul_mul, bias_add, conv2d = info
            activation = ['leaky', leaky_reul_max, leaky_reul_mul]
        elif self.type_match(info, ['Maximum', 'Mul', 'FusedBatchNorm', 'BiasAdd', 'Conv2D']):
            leaky_reul_max, leaky_reul_mul, batch_norm, bias_add, conv2d = info
            activation = ['leaky', leaky_reul_max, leaky_reul_mul]
        elif self.type_match(info, ['Maximum', 'Mul', 'FusedBatchNorm', 'Conv2D']):
            leaky_reul_max, leaky_reul_mul, batch_norm, conv2d = info
            activation = ['leaky', leaky_reul_max, leaky_reul_mul]
        elif self.type_match(info, ['Maximum', 'Mul', 'Merge', 'FusedBatchNorm', 'Switch', 'Conv2D']):
            leaky_reul_max, leaky_reul_mul, merge_1, batch_norm, switch_1, conv2d = info
            activation = ['leaky', leaky_reul_max, leaky_reul_mul]
        elif self.type_match(info, ['Maximum', 'Mul', 'Add', 'Mul', 'Conv2D']):
            leaky_reul_max, leaky_reul_mul, bn_add, bn_mul, conv2d = info
            bn_div, bn_sub = 1, 0
            activation = ['leaky', leaky_reul_max, leaky_reul_mul]
            batch_norm = [bn_add, bn_mul]
        elif self.type_match(info, ['Maximum', 'Mul', 'Add', 'Mul', 'RealDiv', 'Sub', 'Conv2D']):
            leaky_reul_max, leaky_reul_mul, bn_add, bn_mul, bn_div, bn_sub, conv2d = info
            activation = ['leaky', leaky_reul_max, leaky_reul_mul]
            batch_norm = [bn_add, bn_mul, bn_div, bn_sub]
        elif self.type_match(info, ['Relu', 'Add', 'Mul', 'RealDiv', 'Sub', 'Conv2D']):
            activation, bn_add, bn_mul, bn_div, bn_sub, conv2d = info
            batch_norm = [bn_add, bn_mul, bn_div, bn_sub]
        elif self.type_match(info, ['Relu6', 'BiasAdd', 'Conv2D']):
            activation, bias_add, conv2d = info
        elif self.type_match(info, ['Relu6', 'FusedBatchNorm', 'BiasAdd', 'Conv2D']):
            activation, batch_norm, bias_add, conv2d = info
        elif self.type_match(info, ['Relu6', 'FusedBatchNorm', 'Conv2D']):
            activation, batch_norm, conv2d = info
        else:
            raise ValueError(
                'not supported convolutional info. with',
                [node.op.name for node in info], 'as', [node.op.type for node in info]
            )

        self.config['batch_normalize'] = 1 if batch_norm is not None else 0

        self.tensor_conv_w = conv2d.op.inputs[1]
        self.tensor_conv_x = conv2d.op.inputs[0]
        self.tensor_conv_y = conv2d
        # self.tensor_bn = bn_add if bn_add is not None else batch_norm
        if activation is not None:
            self.tensor_activation = activation
        elif batch_norm is not None:
            if isinstance(batch_norm, list):
                self.tensor_activation = bn_add
            else:
                self.tensor_activation = batch_norm
        elif bias_add is not None:
            self.tensor_activation = bias_add

        assert (isinstance(conv2d, tf.Tensor))
        self.config['size'] = int(conv2d.op.inputs[1].shape[0])
        self.config['stride'] = conv2d.op.get_attr('strides')[1]
        self.config['pad'] = 1 if conv2d.op.get_attr('padding') != 'SAME' else 0
        self.config['filters'] = int(conv2d.shape[3])

        if isinstance(activation, list):
            if activation[0] == 'leaky':
                leaky_mul = sess.run(leaky_reul_mul.op.inputs[0], dataset)
                self.config['activation'] = ['leaky', leaky_mul]
            else:
                self.config['activation'] = activation[0]
            self.tensor_activation = activation[1]
        elif activation is not None:
            assert (isinstance(activation, tf.Tensor))
            self.config['activation'] = activation.op.type
        else:
            self.config['activation'] = 'linear'

        self.weights = sess.run(conv2d.op.inputs[1], dataset)
        if bias_add is not None:
            self.bias = sess.run(bias_add.op.inputs[1], dataset)

        if isinstance(batch_norm, list):
            if isinstance(bn_sub, tf.Tensor) and isinstance(bn_div, tf.Tensor):
                self.batch_normalize_moving_mean = sess.run(bn_sub.op.inputs[1], dataset)
                self.batch_normalize_moving_variance = sess.run(bn_div.op.inputs[1].op.inputs[0].op.inputs[0], dataset)
                self.batch_normalize_epsilon = sess.run(bn_div.op.inputs[1].op.inputs[1], dataset)
            else:
                self.batch_normalize_moving_mean = bn_sub
                self.batch_normalize_moving_variance = bn_div
            self.batch_normalize_gamma = sess.run(bn_mul.op.inputs[1], dataset)
            self.batch_normalize_beta = sess.run(bn_add.op.inputs[1], dataset)
        elif batch_norm is not None:
            assert (isinstance(batch_norm, tf.Tensor))
            if 'gamma/read' not in batch_norm.op.inputs[1].name:
                print('[warning] gamma/read should in name:', batch_norm.op.inputs[1].name)
            if 'beta/read' not in batch_norm.op.inputs[2].name:
                print('[warning] beta/read should in name:', batch_norm.op.inputs[2].name)

            self.batch_normalize_gamma = sess.run(batch_norm.op.inputs[1], dataset)
            self.batch_normalize_beta = sess.run(batch_norm.op.inputs[2], dataset)
            if len(batch_norm.op.inputs) == 5:
                if 'moving_mean/read' not in batch_norm.op.inputs[3].name:
                    print('[warning] moving_mean/read should in name:', batch_norm.op.inputs[3].name)
                if 'moving_variance/read' not in batch_norm.op.inputs[4].name:
                    print('[warning] moving_variance/read should in name:', batch_norm.op.inputs[4].name)
                self.batch_normalize_moving_mean = sess.run(batch_norm.op.inputs[3], dataset)
                self.batch_normalize_moving_variance = sess.run(batch_norm.op.inputs[4], dataset)
            else:
                batch_norm_1 = batch_norm.op.outputs[1]
                batch_norm_2 = batch_norm.op.outputs[2]
                batch_normal_outputs = [
                    op for k, op in sess.graph._nodes_by_name.items()
                    if len(op.inputs) == 2 and op.inputs[1] in (batch_norm_1, batch_norm_2)
                ]
                mean_tensor = batch_normal_outputs[0].inputs[0]
                variance_tensor = batch_normal_outputs[1].inputs[0]
                assert ('moving_mean/read' in mean_tensor.name)
                assert ('moving_variance/read' in variance_tensor.name)
                self.batch_normalize_moving_mean = sess.run(mean_tensor, dataset)
                self.batch_normalize_moving_variance = sess.run(variance_tensor, dataset)

            assert (isinstance(self.batch_normalize_moving_mean, np.ndarray))
            if self.batch_normalize_moving_mean.size == 0:
                raise ValueError(
                    'can not find moving_mean values, use is_training=False in {} may help.'.format(batch_norm.name))

            assert (isinstance(self.batch_normalize_moving_variance, np.ndarray))
            if self.batch_normalize_moving_variance.size == 0:
                raise ValueError('can not find moving_variance values, use is_training=False in {} may help.'.format(
                    batch_norm.name))

            self.batch_normalize_epsilon = batch_norm.op.get_attr('epsilon')
            assert (batch_norm.op.get_attr('is_training') == False)


class LayerDepthwiseConvolutional(LayerBase):
    def __init__(self, sess, dataset, info):
        super().__init__()
        self.name = 'depthwise_convolutional'
        self.config = {}
        self.tensor = info
        self.batch_normalize_epsilon = 0
        bias_add = None
        batch_norm = None
        leaky_reul_mul = None
        if self.type_match(info, ['Relu', 'FusedBatchNorm', 'BiasAdd', 'DepthwiseConv2dNative']):
            activation, batch_norm, bias_add, dwconv = info
        elif self.type_match(info, ['Relu', 'BiasAdd', 'DepthwiseConv2dNative']):
            activation, bias_add, dwconv = info
        elif self.type_match(info, ['Relu', 'FusedBatchNorm', 'DepthwiseConv2dNative']):
            activation, batch_norm, dwconv = info
        elif self.type_match(info, ['Relu6', 'BiasAdd', 'DepthwiseConv2dNative']):
            activation, bias_add, dwconv = info
        elif self.type_match(info, ['Relu6', 'FusedBatchNorm', 'BiasAdd', 'DepthwiseConv2dNative']):
            activation, batch_norm, bias_add, dwconv = info
        elif self.type_match(info, ['Relu6', 'FusedBatchNorm', 'DepthwiseConv2dNative']):
            activation, batch_norm, dwconv = info
        elif self.type_match(info, ['LeakyRelu', 'FusedBatchNorm', 'BiasAdd', 'DepthwiseConv2dNative']):
            activation, batch_norm, bias_add, dwconv = info
        elif self.type_match(info, ['Maximum', 'Mul', 'Add', 'Mul', 'RealDiv', 'Sub', 'DepthwiseConv2dNative']):
            leaky_reul_max, leaky_reul_mul, bn_add, bn_mul, bn_div, bn_sub, dwconv = info
            activation = ['leaky', leaky_reul_max, leaky_reul_mul]
            batch_norm = [bn_add, bn_mul, bn_div, bn_sub]
        elif self.type_match(info, ['Relu', 'Add', 'Mul', 'RealDiv', 'Sub', 'DepthwiseConv2dNative']):
            activation, bn_add, bn_mul, bn_div, bn_sub, dwconv = info
            batch_norm = [bn_add, bn_mul, bn_div, bn_sub]
        else:
            raise ValueError(
                'not supported dw_convolutional info. with',
                [node.op.name for node in info], 'as', [node.op.type for node in info]
            )

        self.config['batch_normalize'] = 1 if batch_norm is not None else 0

        self.tensor_conv_w = dwconv.op.inputs[1]
        self.tensor_conv_x = dwconv.op.inputs[0]
        self.tensor_conv_y = dwconv
        self.weights = sess.run(self.tensor_conv_w, dataset)
        # self.tensor_activation = activation or batch_norm or bais_add
        if activation is not None:
            self.tensor_activation = activation
        elif batch_norm is not None:
            self.tensor_activation = batch_norm
        elif bias_add is not None:
            self.tensor_activation = bias_add

        assert (isinstance(dwconv, tf.Tensor))
        self.config['size'] = int(dwconv.op.inputs[1].shape[0])
        self.config['stride'] = dwconv.op.get_attr('strides')[1]
        self.config['pad'] = 1 if dwconv.op.get_attr('padding') != 'SAME' else 0

        if isinstance(activation, list):
            if activation[0] == 'leaky':
                leaky_mul = sess.run(leaky_reul_mul.op.inputs[0], dataset)
                self.config['activation'] = ['leaky', leaky_mul]
            else:
                self.config['activation'] = activation[0]
            self.tensor_activation = activation[1]
        elif activation is not None:
            assert (isinstance(activation, tf.Tensor))
            self.config['activation'] = activation.op.type
        else:
            self.config['activation'] = 'linear'

        self.bias = sess.run(bias_add.op.inputs[1], dataset) if bias_add is not None else None

        if isinstance(batch_norm, list):
            bn_add, bn_mul, bn_div, bn_sub = batch_norm
            self.batch_normalize_moving_mean = sess.run(bn_sub.op.inputs[1], dataset)
            self.batch_normalize_moving_variance = sess.run(bn_div.op.inputs[1].op.inputs[0].op.inputs[0], dataset)
            self.batch_normalize_epsilon = sess.run(bn_div.op.inputs[1].op.inputs[1], dataset)
            self.batch_normalize_gamma = sess.run(bn_mul.op.inputs[1], dataset)
            self.batch_normalize_beta = sess.run(bn_add.op.inputs[1], dataset)
        elif batch_norm is not None:
            assert (isinstance(batch_norm, tf.Tensor))
            if 'gamma/read' not in batch_norm.op.inputs[1].name:
                print('[warning] gamma/read should in name:', batch_norm.op.inputs[1].name)
            if 'beta/read' not in batch_norm.op.inputs[2].name:
                print('[warning] beta/read should in name:', batch_norm.op.inputs[2].name)
            self.batch_normalize_gamma = sess.run(batch_norm.op.inputs[1], dataset)
            self.batch_normalize_beta = sess.run(batch_norm.op.inputs[2], dataset)
            if len(batch_norm.op.inputs) == 5:
                if 'moving_mean/read' not in batch_norm.op.inputs[3].name:
                    print('[warning] moving_mean/read should in name:', batch_norm.op.inputs[3].name)
                if 'moving_variance/read' not in batch_norm.op.inputs[4].name:
                    print('[warning] moving_variance/read should in name:', batch_norm.op.inputs[4].name)
                self.batch_normalize_moving_mean = sess.run(batch_norm.op.inputs[3], dataset)
                self.batch_normalize_moving_variance = sess.run(batch_norm.op.inputs[4], dataset)
            else:
                batch_norm_1 = batch_norm.op.outputs[1]
                batch_norm_2 = batch_norm.op.outputs[2]
                batch_normal_outputs = [
                    op for k, op in sess.graph._nodes_by_name.items()
                    if len(op.inputs) == 2 and op.inputs[1] in (batch_norm_1, batch_norm_2)
                ]
                mean_tensor = batch_normal_outputs[0].inputs[0]
                variance_tensor = batch_normal_outputs[1].inputs[0]
                assert ('moving_mean/read' in mean_tensor.name)
                assert ('moving_variance/read' in variance_tensor.name)
                self.batch_normalize_moving_mean = sess.run(mean_tensor, dataset)
                self.batch_normalize_moving_variance = sess.run(variance_tensor, dataset)

            self.batch_normalize_epsilon = batch_norm.op.get_attr('epsilon')
            assert (batch_norm.op.get_attr('is_training') == False)


class LayerPool(LayerBase):
    def __init__(self, sess, info):
        super().__init__()
        self.config = {}
        self.tensor_pool = info[0]
        if self.tensor_pool.op.type not in ('MaxPool', 'AvgPool'):
            raise 'not supported pooling {}'.format(self.tensor_pool.op.type)

        assert (isinstance(self.tensor_pool, tf.Tensor))
        self.config['size'] = self.tensor_pool.op.get_attr('ksize')[1]
        self.config['stride'] = self.tensor_pool.op.get_attr('strides')[1]


def convert_layer(sess, dataset, info):
    ty = info[0]
    info = info[1:]
    if ty == 'net':
        return LayerNet(sess, info)
    elif ty == 'convolutional':
        return LayerConvolutional(sess, dataset, info)
    elif ty == 'depthwise_convolutional':
        return LayerDepthwiseConvolutional(sess, dataset, info)
    elif ty == 'pool':
        return LayerPool(sess, info)
    else:
        raise ValueError('unknown type:', ty)


def convert_to_layers(sess, dataset, info_list):
    info_list.reverse()
    return [convert_layer(sess, dataset, info) for info in info_list]
