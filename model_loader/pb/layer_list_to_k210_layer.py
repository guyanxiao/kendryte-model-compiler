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

import k210_layer
import numpy as np
from . import tensor_list_to_layer_list


def make_k210_layer_from_tensor(sess, dataset, buffer, input_min, input_max, eight_bit_mode, range_from_batch, idx):
    pool_tensor_info = dict()
    pool_type_size_stride = None  # bypass pool

    if isinstance(buffer[-1], tensor_list_to_layer_list.LayerConvolutional) \
            or isinstance(buffer[-1], tensor_list_to_layer_list.LayerDepthwiseConvolutional):
        conv_layer = buffer.pop()

        input_shape = list(sess.run(conv_layer.tensor_conv_x, dataset).shape)
        conved_shape = list(sess.run(conv_layer.tensor_conv_y, dataset).shape)
        output_shape = conved_shape

        if conv_layer.tensor_conv_x.op.type == 'SpaceToBatchND':
            print('[warning] found SpaceToBatchND fix input_shape/')
            input_shape[1] = input_shape[1]-2
            input_shape[2] = input_shape[2]-2

        weights_min, weights_max, _ = range_from_batch(sess, conv_layer.tensor_conv_w, dataset, is_weights=True)
        conv_weights = conv_layer.weights
        conv_isdw = isinstance(conv_layer, tensor_list_to_layer_list.LayerDepthwiseConvolutional)
        conv_tensor_info = {'name': conv_layer.tensor_conv_y.name}

        if int(conv_layer.config['batch_normalize']) == 1:
            bn_mean_var_gamma_beta_epsilon = [
                conv_layer.batch_normalize_moving_mean - (conv_layer.bias if conv_layer.bias is not None else 0),
                conv_layer.batch_normalize_moving_variance,
                conv_layer.batch_normalize_gamma,
                conv_layer.batch_normalize_beta,
                conv_layer.batch_normalize_epsilon
            ]
            bn_tensor_info = {'name': 'bn'}
        else:
            if conv_layer.bias is not None:
                bias_shape = conv_layer.bias.shape
                bn_mean_var_gamma_beta_epsilon = [
                    0, 1, np.ones(bias_shape), conv_layer.bias, 0
                ]
            else:
                bn_mean_var_gamma_beta_epsilon = [
                    0, 1, np.ones([conved_shape[3]]), 0, 0
                ]

            bn_tensor_info = {'name': 'bn'}

        tensor_act = conv_layer.tensor_activation
        act_min_y, act_max_y, _ = range_from_batch(sess, tensor_act, dataset)
        act_type = conv_layer.config['activation']
        act_tensor_info = {'name': tensor_act.name if tensor_act is not None else 'default_linear'}
    else:
        raise ValueError('unsupported type seq: ', *[type(l) for l in buffer])

    if len(buffer) > 0 and isinstance(buffer[-1], tensor_list_to_layer_list.LayerPool):
        pool_layer = buffer.pop()
        assert (isinstance(pool_layer, tensor_list_to_layer_list.LayerPool))
        pool_size = pool_layer.config['size']
        pool_stride = pool_layer.config['stride']
        pool_type = pool_layer.tensor_pool.op.type

        if pool_size == 2 and pool_layer.tensor_pool.op.inputs[0].shape[3] % 2 != 0:
            if pool_layer.tensor_pool.op.get_attr('padding') == b'SAME':
                raise ValueError("at {} unsupport padding mode SAME of pooling with size == 2" \
                                 .format(pool_layer.tensor_pool.name))

        pool_type_size_stride = [pool_type, pool_size, pool_stride]
        pool_tensor_info = {'name': pool_layer.tensor_pool.op.name}
        output_shape = [output_shape[0], output_shape[1]//pool_stride, output_shape[2]//pool_stride, output_shape[3]]

    return {
        'iwo_minmax':[input_min, input_max, weights_min, weights_max, act_min_y, act_max_y],
        'ico_shapes':[input_shape, conved_shape, output_shape],
        'conv_weights_isdw':[conv_weights, conv_isdw],
        'bn_mean_var_gamma_beta_epsilon':bn_mean_var_gamma_beta_epsilon,
        'act_type':act_type,
        'pool_type_size_stride':pool_type_size_stride,
        'eight_bit_mode':eight_bit_mode,
        'cbap_tensor_info':[conv_tensor_info, bn_tensor_info, act_tensor_info, pool_tensor_info],
        'idx':idx
    }


def gen_k210_layers(layers: [tensor_list_to_layer_list.LayerBase], sess, dataset, range_from_batch,
                    eight_bit_mode=False, input_min=0, input_max=1, layer_start_idx=0):
    buffer = list(layers)
    buffer.reverse()
    kl_args_list = []

    net = buffer.pop()
    assert (isinstance(net, tensor_list_to_layer_list.LayerNet))

    while len(buffer) != 0:
        if len(kl_args_list) > 0:
            last_min = kl_args_list[-1]['iwo_minmax'][4]
            last_max = kl_args_list[-1]['iwo_minmax'][5]
        else:
            last_min = input_min
            last_max = input_max

        cur_k210_arg = make_k210_layer_from_tensor(
            sess=sess, dataset=dataset,
            buffer=buffer,
            input_min=last_min, input_max=last_max,
            eight_bit_mode=eight_bit_mode,
            range_from_batch=range_from_batch,
            idx=len(kl_args_list)+layer_start_idx
        )
        kl_args_list.append(cur_k210_arg)

    kl_args_fixed = k210_layer.k210_layer_post_fix(kl_args_list)
    kl_list = [k210_layer.K210Layer(**kl_args) for kl_args in kl_args_fixed]
    return kl_list
