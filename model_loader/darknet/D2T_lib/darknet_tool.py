from struct import unpack, pack
import numpy as np
from . import bwidth
from . import net

# C-types

__unpack_types__ = {
    'float32': 'f',
    'float64': 'd',
    'float': 'f',
    'double': 'd',
    'int8': 'b',
    'int16': 'i',
    'int32': 'l',
    'uint8': 'B',
    'uint16': 'I',
    'uint32': 'L'
}

"""
darknet_network: storing network structure & parameters.
interference features:
- storing virtual network with <D2T_lib/net> class
- import .weights -> encoded/decoded parameters
- tools for modifying parameters (suspended)

- export self -> .weights (x)
- export self -> .cfg (by <D2T_lib/net>) (x)
"""

# count of total dartknet-nets
__count_darknet__ = 0

# decode order
__decode_order__ = ['bias', 'bn_scale', 'bn_mean', 'bn_variance', 'weights']


class darknet_network(object):
    def __init__(self, name=None,
                 cfg_file=None, weights_file=None,
                 dtype='float32'):
        global __count_darknet__
        self.name = name if name else 'darknet_%i' % (__count_darknet__)
        __count_darknet__ += 1

        # virtual network structure
        self.net = net.net(name=self.name, dtype=dtype)

        # layer parameters
        # buffer encoded: not to decode
        self.raw_buf = {}
        # buffer decoded
        self.dec_buf = {}

        # data info
        self.dtype = dtype
        self.byte_per = bwidth.__bwidth__[dtype]

        # if has net structure
        self.__own_cfg__ = False
        self.__own_weights__ = False
        if cfg_file:
            if not name:
                self.name = cfg_file.split('/')[-1].split('.cfg')[0]
            self.__own_cfg__ = True
            self.from_cfg_file(cfg_file)
            __count_darknet__ -= 1
            if weights_file:
                self.from_weights_file(weights_file)
                self.__own_weights__ = True

    def from_cfg_file(self, cfg_file):
        with open(cfg_file, 'r') as F:
            self.net.layers_from_cfg(cfg_file)

    def from_weights_file(self, weights_file):
        bw = bwidth.__bwidth__[self.dtype]
        with open(weights_file, 'rb') as F:
            # header
            self.raw_buf['header'] = F.read(4 + 4 + 4 + 8)
            if self.__own_cfg__:
                # read layer-by-layer
                for i in self.net.route:
                    _layer = self.net.layers[i]
                    jump_len = bw * sum(_layer.my_size('count').values())
                    self.raw_buf[_layer.name] = F.read(jump_len)

            self.raw_buf['raw_rest'] = F.read(-1)  # all the rest

    def decode_buf(self, buf_id, shape_as=None):
        # bias, (scale,mean,var), weights
        _layer = self.net.layers[buf_id]
        size_info = _layer.my_size('count')
        buf_dict = dict()

        bw = bwidth.__bwidth__[self.dtype]

        now_at = 0
        for k in __decode_order__:
            v = size_info[k] if k in size_info else 0
            if v > 0:  # valid space
                # decoding with fixed byte-width
                _raw_buf = self.raw_buf[_layer.name]
                print('len raw_buf:', _layer.name, len(_raw_buf))
                skip_len = v * bw
                buf_dict[k] = np.array(
                    unpack('%i%s' % (v, __unpack_types__[self.dtype]),
                           _raw_buf[now_at: now_at + skip_len]),
                    dtype=self.dtype)
                # reshape weights as tf variables
                now_at += skip_len

        if shape_as and size_info['weights'] > 0:
            assert type(shape_as[0]) == tuple, \
                "please pack the resize route into a tuple/list, with each member as a tuple."
            arr = buf_dict['weights']
            if len(shape_as) > 1:
                new_size = shape_as[0]
                arr = np.resize(arr, new_size)
            buf_dict['weights'] = np.transpose(arr, shape_as[-1])

        self.dec_buf[_layer.name] = buf_dict

    def encode_buf(self, buf_id, dtype=None):
        # support encoding into difference dtype
        _dtype = dtype if dtype else self.dtype
        # check the new dtype
        assert _dtype in __unpack_types__, 'unsupported data type: %s' % (_dtype)

        # flatten and encode
        enc_buf = self.dec_buf[buf_id]
        data_arr = []
        for k in __decode_order__:
            if k in enc_buf:
                v = enc_buf[k]
                data_list = np.ravel(v)
                total_number = np.product(v.shape)
                data_arr.append(pack('%i%s' % (total_number, __unpack_types__[_dtype]),
                                     *data_list))
        return data_arr

    def export_weights_to(self, weights_path):
        with open(weights_path, 'wb') as F:
            F.write(self.raw_buf['header'])
            if self.__own_cfg__:
                # write layer-by-layer
                for i in self.net.route:
                    _layer_name = self.net.layers[i].name
                    F.write(self.raw_buf[_layer_name])

            F.write(self.raw_buf['raw_rest'])
