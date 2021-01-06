# coding=utf-8
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

import os
import tempfile
import tensorflow as tf
from .D2T_lib import darknet_tool, tensorflow_tool


def decode_darknet(cfg_file, weights_file, output_dir):
    net1 = darknet_tool.darknet_network('network',
                                        cfg_file=cfg_file,
                                        weights_file=weights_file,
                                        dtype='float32')

    net1.net.statistcs_size(print_out=True)

    if os.path.exists(output_dir) and False:
        print('For security, I won\'t overwrite the existed directory.')
    else:
        tensorflow_tool.darknet_to_tf_module(net1, out_dir=output_dir)


def darknet2pb(input_dir, output_pb_path='tf_model', input_node_name='input'):
    import tools
    data_loader = tools.import_from_path(os.path.join(input_dir, 'network.py'))
    data_loader.load_data()

    # get basic info
    info_dict = dict()
    with open(os.path.join(input_dir, 'info.txt'), 'r') as F_info:
        contents = F_info.readlines()
        for l in contents:
            if len(l) > 1 and l[0] != '#':
                k, v = l.strip().split(':')
                info_dict[k.strip()] = v.strip()
    input_w = (int)(info_dict['width'])
    input_h = (int)(info_dict['height'])
    channel = (int)(info_dict['channel'])
    dtype = info_dict['data type']

    # set input placeholder
    inp = tf.placeholder(shape=[None, input_h, input_w, channel],
                         dtype=dtype,
                         name=input_node_name)

    yv2 = data_loader.network_forward(inp)

    with open(os.path.join(input_dir, 'info.txt'), 'a') as F_info:
        F_info.write('\n#I/O info\n')
        F_info.write('input node: {}\n'.format(input_node_name.split(':')[0]))
        F_info.write('output node: {}\n'.format(yv2.name.split(':')[0]))

    # # Tensorflow session
    sess1 = tf.Session()
    sess1.run(tf.global_variables_initializer())

    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = convert_variables_to_constants(sess1, sess1.graph_def, [yv2.name.split(':')[0]])
    tf.train.write_graph(graph, input_dir, '{}.pb'.format(output_pb_path), as_text=False)
    tf.reset_default_graph()

    return inp.name, yv2.op.name, input_w, input_h


def load_model(dataset_val, range_from_batch, args):
    import tools
    pb_loader = tools.import_from_path(os.path.dirname(__file__) + '/../pb')

    cfg_path = args.cfg_path
    weights_path = args.weights_path
    build_dir = tempfile.mkdtemp()
    decode_darknet(cfg_path, weights_path, build_dir)
    pb_name = 'output'
    dataset_input_name, tensor_output_name, input_w, input_h = darknet2pb(build_dir, pb_name, 'input')
    assert (args.image_w == input_w)
    assert (args.image_h == input_h)

    pb_path = os.path.join(build_dir, pb_name + '.pb')

    args.pb_path = pb_path
    args.tensor_output_name = tensor_output_name
    args.dataset_input_name = dataset_input_name

    return pb_loader.load_model(dataset_val, range_from_batch, args)
