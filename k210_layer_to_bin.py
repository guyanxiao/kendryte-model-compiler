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
import math
from struct import pack
import tools

default_conv_arg = None
default_act_arg = None
default_bn_arg = {
    'load_para': 0,
    'bwsx_base_addr': 0
}
default_pool_arg = {
    'pool_type': 0,  # bypass
}

kpu_layer_config_reg_offset = {
    'interrupt_enabe': 0,
    'image_addr': 1,
    'image_channel_num': 2,
    'image_size': 3,
    'kernel_pool_type_cfg': 4,
    'kernel_load_cfg': 5,
    'kernel_offset': 6,
    'kernel_calc_type_cfg': 7,
    'write_back_cfg': 8,
    'conv_value': 9,
    'conv_value2': 10,
    'dma_parameter': 11
}

kpu_layer_config_field_offset = {
    'interrupt_enabe': {
        'int_en': 0,
        'ram_flag': 1,
        'full_add': 2,
        'depth_wise_layer': 3
    },
    'image_addr': {
        'image_src_addr': 0,
        'image_dst_addr': 32
    },
    'image_channel_num': {
        'i_ch_num': 0,
        'o_ch_num': 32,
        'o_ch_num_coef': 48
    },
    'image_size': {
        'i_row_wid': 0,
        'i_col_high': 10,
        'o_row_wid': 32,
        'o_col_high': 42
    },
    'kernel_pool_type_cfg': {
        'kernel_type': 0,
        'pad_type': 3,
        'pool_type': 4,
        'first_stride': 8,
        'bypass_conv': 9,
        'load_para': 10,
        'dma_burst_size': 16,
        'pad_value': 24,
        'bwsx_base_addr': 32
    },
    'kernel_load_cfg': {
        'load_coor': 0,
        'load_time': 1,
        'para_size': 15,
        'para_start_addr': 32
    },
    'kernel_offset': {
        'coef_column_offset': 0,
        'coef_row_offset': 4
    },
    'kernel_calc_type_cfg': {
        'channel_switch_addr': 0,
        'row_switch_addr': 16,
        'coef_size': 20,
        'coef_group': 28,
        'load_act': 31,
        'active_addr': 32
    },
    'write_back_cfg': {
        'wb_channel_switch_addr': 0,
        'wb_row_switch_addr': 16,
        'wb_group': 20
    },
    'conv_value': {
        'shr_w': 0,
        'shr_x': 4,
        'arg_w': 8,
        'arg_x': 32
    },
    'conv_value2': {
        'arg_add': 0
    },
    'dma_parameter': {
        'send_data_out': 0,
        'channel_byte_num': 16,
        'dma_total_byte': 32
    }
}


def signed_to_hex(value, width):
    return hex(int(round((1 << width) + value)) % (1 << width))


def min_max_to_scale_bias(minv, maxv):
    scale = (maxv - minv) / 255
    bias = minv
    return scale, bias


def gen_layer_struct(klayer: k210_layer.K210Layer, idx: int):
    reserved = 0
    set_to_zero = 0
    img_ram_size = 2 * 1024 * 1024

    conv_arg = klayer.conv and klayer.conv.to_k210() or default_conv_arg
    bn_arg = klayer.bn and klayer.bn.to_k210(conv_arg['swsx']) or default_bn_arg
    act_arg = klayer.act and klayer.act.to_k210(bn_arg['post_scale']) or default_act_arg
    pool_arg = klayer.pool and klayer.pool.to_k210() or default_pool_arg
    io_arg = klayer.to_k210()

    mino, maxo = klayer.act.min_y, klayer.act.max_y
    output_scale, output_bias = tools.min_max_to_scale_bias(mino, maxo)

    img_input_size = int(math.ceil(io_arg['i_ch_num'] / conv_arg['coef_group']) * 64 * conv_arg['channel_switch_addr'])
    img_output_size = int(math.ceil(io_arg['o_ch_num'] / io_arg['wb_group']) * 64 * io_arg['wb_channel_switch_addr'])

    assert (img_input_size + img_output_size <= img_ram_size)

    interrupt_enabe = {
        'int_en': set_to_zero,
        'ram_flag': reserved,
        'full_add': set_to_zero,
        'depth_wise_layer': conv_arg['depth_wise_layer']
    }
    image_addr = {
        'image_src_addr': hex(int((0 if not idx & 1 else (img_ram_size - img_input_size)) / 64)),
        'image_dst_addr': hex(int((0 if idx & 1 else (img_ram_size - img_output_size)) / 64))
    }
    image_channel_num = {
        'i_ch_num': hex(io_arg['i_ch_num'] - 1),
        'o_ch_num': hex(io_arg['o_ch_num'] - 1),
        'o_ch_num_coef': hex(conv_arg['o_ch_num_coef'] - 1),
    }
    image_size = {
        'i_row_wid': hex(conv_arg['i_row_wid'] - 1),
        'i_col_high': hex(conv_arg['i_col_high'] - 1),
        'o_row_wid': hex(io_arg['o_row_wid'] - 1),
        'o_col_high': hex(io_arg['o_col_high'] - 1),
    }
    kernel_pool_type_cfg = {
        'kernel_type': conv_arg['kernel_type'],
        'pad_type': conv_arg['pad_type'],
        'pool_type': pool_arg['pool_type'],
        'first_stride': conv_arg['first_stride'],
        'bypass_conv': 0 if klayer.conv else 1,
        'load_para': bn_arg['load_para'],
        'dma_burst_size': io_arg['dma_burst_size'],
        'pad_value': signed_to_hex(conv_arg['pad_value'], 8),
        'bwsx_base_addr': bn_arg['bwsx_base_addr'],
    }
    kernel_load_cfg = {
        'load_coor': conv_arg['load_coor'],
        'load_time': conv_arg['load_time'] - 1,
        'para_size': conv_arg['para_size'],
        'para_start_addr': conv_arg['para_start_addr'],
    }
    kernel_offset = {
        'coef_column_offset': set_to_zero,
        'coef_row_offset': set_to_zero,
    }
    kernel_calc_type_cfg = {
        'channel_switch_addr': hex(conv_arg['channel_switch_addr']),
        'row_switch_addr': hex(conv_arg['row_switch_addr']),
        'coef_size': reserved,
        'coef_group': conv_arg['coef_group'],
        'load_act': 1 if klayer.act else 0,
        'active_addr': act_arg['active_addr']
    }
    write_back_cfg = {
        'wb_channel_switch_addr': hex(io_arg['wb_channel_switch_addr']),
        'wb_row_switch_addr': hex(io_arg['wb_row_switch_addr']),
        'wb_group': io_arg['wb_group']
    }
    conv_value = {
        'shr_w': conv_arg['shr_w'],
        'shr_x': conv_arg['shr_x'],
        'arg_w': signed_to_hex(conv_arg['arg_w'], 24),
        'arg_x': signed_to_hex(conv_arg['arg_x'], 24),
    }
    conv_value2 = {
        'arg_add': int(round(conv_arg['arg_add'])),
    }
    dma_parameter = {
        'send_data_out': io_arg['send_data_out'],
        'channel_byte_num': io_arg['channel_byte_num'] - 1,
        'dma_total_byte': io_arg['dma_total_byte'] - 1,
    }

    return {
        'interrupt_enabe': interrupt_enabe,
        'image_addr': image_addr,
        'image_channel_num': image_channel_num,
        'image_size': image_size,
        'kernel_pool_type_cfg': kernel_pool_type_cfg,
        'kernel_load_cfg': kernel_load_cfg,
        'kernel_offset': kernel_offset,
        'kernel_calc_type_cfg': kernel_calc_type_cfg,
        'write_back_cfg': write_back_cfg,
        'conv_value': conv_value,
        'conv_value2': conv_value2,
        'dma_parameter': dma_parameter
    }, (output_scale, output_bias)


def gen_layer_list_struct(klayers: [k210_layer.K210Layer]):
    ret = [
        gen_layer_struct(klayer, idx)
        for klayer, idx in zip(klayers, range(len(klayers)))
    ]
    return ret


def gen_layer_code(dlayer, layer_cfg):
    layer_cfg.reg_arg = bytearray(len(dlayer[0].items()) * 8)
    for reg_name, data in dlayer[0].items():
        value = 0
        for filed_name, filed_value in data.items():
            if isinstance(filed_value, int):
                pass
            elif isinstance(filed_value, str):
                filed_value = int(filed_value, 16) if '0x' in filed_value else int(filed_value)
            else:
                filed_value = 0
            value += (filed_value << kpu_layer_config_field_offset[reg_name][filed_name])
        layer_cfg.reg_arg[kpu_layer_config_reg_offset[reg_name]*8:kpu_layer_config_reg_offset[reg_name]*8+8] = value.to_bytes(8, 'little')


def gen_bn_code(dlayer, layer_cfg):
    bn_list = dlayer[0]['kernel_pool_type_cfg']['bwsx_base_addr']
    layer_cfg.bn_len = len(bn_list) * 8
    layer_cfg.bn_arg = bytearray(layer_cfg.bn_len)
    i = 0
    for bn in bn_list:
            layer_cfg.bn_arg[i:i+8] = (int(bn['norm_mul'], 16) + (int(bn['norm_add'], 16) << 24) + (int(bn['norm_shift']) << 56)).to_bytes(8, 'little')
            i += 8

def gen_act_code(dlayer, layer_cfg):
    act_list = dlayer[0]['kernel_calc_type_cfg']['active_addr']
    for item in act_list:
        layer_cfg.act_arg += (int(item['dxs']) + (int(item['dy']) << 8) + (int(signed_to_hex(item['x'], 36), 16) << 24)).to_bytes(8, 'little')
    bias_list = [int(item['y']) for item in act_list]
    value1, value2 = 0, 0
    for index in range(8):
        value1 += (bias_list[index] << (8 * index))
        value2 += (bias_list[index + 8] << (8 * index))
    layer_cfg.act_arg += value1.to_bytes(8, 'little')
    layer_cfg.act_arg += value2.to_bytes(8, 'little')


def gen_weights_code(dlayer, layer_cfg, eight_bit_mode):
    weights = dlayer[0]['kernel_load_cfg']['para_start_addr']
    if eight_bit_mode:
        layer_cfg.weights_len = len(weights)
        layer_cfg.weights_arg = bytearray(layer_cfg.weights_len)
        i = 0
        for item in weights:
            layer_cfg.weights_arg[i:i+1] = int(signed_to_hex(item, 8), 16).to_bytes(1, 'little')
            i += 1
    else:
        layer_cfg.weights_len = len(weights) * 2
        layer_cfg.weights_arg = bytearray(layer_cfg.weights_len)
        i = 0
        for item in weights:
            layer_cfg.weights_arg[i:i+2] = int(signed_to_hex(item, 16), 16).to_bytes(2, 'little')
            i += 2


class layer_config_struct():
    def __init__(self):
        self.reg_addr_offset = 0
        self.reg_arg = b''
        self.act_addr_offset = 0
        self.act_arg = b''
        self.bn_addr_offset = 0
        self.bn_len = 0
        self.bn_arg = b''
        self.weights_addr_offset = 0
        self.weights_len = 0
        self.weights_arg = b''


def gen_layer_bin(klayers: [k210_layer.K210Layer], eight_bit_mode):
    structs = gen_layer_list_struct(klayers)
    output_scale, output_bias = structs[-1][1]

    model_config_length = 64
    magic_number = 0x12345678
    layer_number = len(structs)
    model_config_part = int(magic_number).to_bytes(4, 'little') + \
                        int(layer_number).to_bytes(4, 'little') + \
                        int(model_config_length).to_bytes(4, 'little') + \
                        int(eight_bit_mode).to_bytes(4, 'little') + \
                        pack('<f', output_scale) + \
                        pack('<f', output_bias)
    if len(model_config_part) > model_config_length:
        print('model config param error, please check')
        raise  TypeError(AttributeError)
    model_config_part += bytearray(model_config_length - len(model_config_part))
   
    layer_config = [layer_config_struct() for x in range(layer_number)]

    for index in range(layer_number):
        gen_layer_code(structs[index], layer_config[index])
        gen_act_code(structs[index], layer_config[index])
        gen_bn_code(structs[index], layer_config[index])
        gen_weights_code(structs[index], layer_config[index], eight_bit_mode)

    layer_config_length = 24 * layer_number
    layer_config[0].reg_addr_offset = model_config_length + layer_config_length
    layer_config[0].act_addr_offset = layer_config[0].reg_addr_offset + 12 * 8
    layer_config[0].bn_addr_offset = layer_config[0].act_addr_offset + 18 * 8
    layer_config[0].weights_addr_offset = layer_config[0].bn_addr_offset + layer_config[0].bn_len
    for index in range(1, layer_number):
        layer_config[index].reg_addr_offset = layer_config[index - 1].weights_addr_offset + layer_config[index - 1].weights_len
        layer_config[index].act_addr_offset = layer_config[index].reg_addr_offset + 12 * 8
        layer_config[index].bn_addr_offset = layer_config[index].act_addr_offset + 18 * 8
        layer_config[index].weights_addr_offset = layer_config[index].bn_addr_offset + layer_config[index].bn_len

    layer_config_part = b''
    param_part = b''
    for layer in layer_config:
        layer_config_part += int(layer.reg_addr_offset).to_bytes(4, 'little')
        layer_config_part += int(layer.act_addr_offset).to_bytes(4, 'little')
        layer_config_part += int(layer.bn_addr_offset).to_bytes(4, 'little')
        layer_config_part += int(layer.bn_len).to_bytes(4, 'little')
        layer_config_part += int(layer.weights_addr_offset).to_bytes(4, 'little')
        layer_config_part += int(layer.weights_len).to_bytes(4, 'little')
        param_part += layer.reg_arg
        param_part += layer.act_arg
        param_part += layer.bn_arg
        param_part += layer.weights_arg

    model_bin = model_config_part + layer_config_part + param_part
    if len(model_bin) % 4:
        model_bin += bytearray(4 - len(model_bin) % 4)
    model_bin_reverse = bytearray(len(model_bin))
    for i in range(0, len(model_bin), 4):
        model_bin_reverse[i] = model_bin[i+3]
        model_bin_reverse[i+1] = model_bin[i+2]
        model_bin_reverse[i+2] = model_bin[i+1]
        model_bin_reverse[i+3] = model_bin[i]

    return model_bin_reverse
