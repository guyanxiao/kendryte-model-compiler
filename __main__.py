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

import argparse
import os

import tensorflow as tf

import k210_layer_to_c_code
import k210_layer_to_bin
import range_from_batch
import tools




# def convert(model_loader, dataset, args):
#     k210_layers = model_loader.load_model(dataset, args)
#
#     output_code = k210_layer_to_c_code.gen_layer_list_code(k210_layers, args.eight_bit_mode, args.prefix,
#                                                            args.layer_start_idx)
#     try:
#         output_bin = k210_layer_to_bin.gen_layer_bin(k210_layers, args.eight_bit_mode)
#     except Exception as e:
#         print(e)
#         output_bin = None
#
#     return (output_code, output_bin)


def main():
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_input_name', default='input:0')
    parser.add_argument('--eight_bit_mode', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--output_path', default='build/gencode_output')
    parser.add_argument('--output_bin_name', default='build/model.bin')
    parser.add_argument('--prefix', default='')
    parser.add_argument('--layer_start_idx', type=int, default=0)

    parser.add_argument('--model_loader', default='model_loader/pb')
    parser.add_argument('--tensorboard_mode', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--pb_path', default=None)
    parser.add_argument('--h5_path', default=None)
    parser.add_argument('--h5_custom_objects', default=None)
    parser.add_argument('--cfg_path', default=None)
    parser.add_argument('--weights_path', default=None)
    parser.add_argument('--tensor_input_name', default=None)
    parser.add_argument('--tensor_output_name', default=None)
    parser.add_argument('--tensor_input_min', type=float, default=0)
    parser.add_argument('--tensor_input_max', type=float, default=1)
    parser.add_argument('--tensor_input_minmax_auto', type=str2bool, nargs='?', const=True, default=True)

    parser.add_argument('--dataset_loader', default='dataset_loader/img_0_1.py')
    parser.add_argument('--dataset_pic_path', default='dataset/yolo')
    parser.add_argument('--dataset_path', default='dataset/yolo')
    parser.add_argument('--image_w', type=int, default=320)
    parser.add_argument('--image_h', type=int, default=240)


    args = parser.parse_args()

    eight_bit_mode = args.eight_bit_mode
    output_path = args.output_path
    output_bin_name = args.output_bin_name
    args.prefix = args.prefix if len(args.prefix) > 0 \
        else os.path.basename(args.output_path).replace('.', '_').replace('-', '_')

    layer_start_idx = args.layer_start_idx

    model_loader = args.model_loader
    tensorboard_mode = args.tensorboard_mode  # used in model loader
    pb_path = args.pb_path  # used in model loader
    tensor_input_name = args.tensor_input_name  # used in model loader
    tensor_output_name = args.tensor_output_name  # used in model loader
    input_min = args.tensor_input_min  # used in model loader
    input_max = args.tensor_input_max  # used in model loader
    input_minmax_auto = args.tensor_input_minmax_auto  # used in model loader

    dataset_loader = args.dataset_loader
    dataset_input_name = args.dataset_input_name
    dataset_pic_path = args.dataset_pic_path  # used in dataset loader
    image_w = args.image_w  # used in dataset loader
    image_h = args.image_h  # used in dataset loader

    if ':' not in dataset_input_name:
        args.dataset_input_name = dataset_input_name + ':0'

    if output_path.endswith('.c'):
        output_path = output_path[:-2]

    dataset_loader_module = tools.import_from_path(dataset_loader)

    dataset_val = dataset_loader_module.load_dataset(args)

    model_loader_module = tools.import_from_path(model_loader)
    rfb = range_from_batch.RangeFromBatchMinMax()
    k210_layers = model_loader_module.load_model(dataset_val, rfb, args)

    c_file, h_file = k210_layer_to_c_code.gen_layer_list_code(
        k210_layers, args.eight_bit_mode, args.prefix, args.layer_start_idx
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path + '.c', 'w') as of:
        of.write(c_file)
    print('generate {} finish'.format(output_path + '.c'))

    with open(output_path + '.h', 'w') as of:
        of.write(h_file)
    print('generate {} finish'.format(output_path + '.h'))

    try:
        output_bin = k210_layer_to_bin.gen_layer_bin(k210_layers, args.eight_bit_mode)
        os.makedirs(os.path.dirname(output_bin_name), exist_ok=True)
        with open(output_bin_name, 'wb') as of:
            of.write(output_bin)
        print('generate bin finish')
    except Exception as e:
        print(e)


if __name__ == '__main__':
    main()
