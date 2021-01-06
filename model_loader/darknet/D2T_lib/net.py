import copy
import re
from . import bwidth
from . import layers

__parse_layers__ = {
    'depthwise_convolutional': layers.dw_convolutional_layer,
    'convolutional': layers.convolutional_layer,
    'connected': None,
    'maxpool': layers.maxpooling_layer,
    'avgpool': layers.avgpooling_layer,
    'route': layers.route_layer,
    'softmax': None
}

__unit_size__ = {
    'bit': 8,
    'byte': 1,
    'KB': 1.0 / 1024,
    'MB': 1.0 / 1024 / 1024,
    'GB': 1.0 / 1024 / 1024 / 1024
}

__empty_info__ = re.compile(' ')
"""
virtual class for only storing the structure of the network without param.

interference features:
- import .cfg -> virtual net
- export self -> .cfg
- export self -> tensorflow code (module like)
- statistics of parameters
"""


class net(object):
    def __init__(self, name, scope=None, dtype='float32'):
        self.name = name
        self.scope = scope
        self.dtype = dtype
        self.layers = dict()

        # input size
        self.input_size = {
            'batch': 1,
            'size': [0],
        }

        # net route : to be used in the future, now as sequence
        self.route = []

        # statistics table
        self.statistics = {}

    def layers_from_cfg(self, cfg_path):
        with open(cfg_path) as F:
            # skip empty lines
            contents = list(l.strip() for l in filter(lambda x: True if len(x) > 1 else False,
                                                      F.readlines()))
            # get blocks for parsing
            block_st = -1
            line_id = 0
            block_id = -1
            for l in contents:
                if l[0] == '[':  # find block header
                    if block_st > -1:
                        # return kwargs for construct layer
                        self.parse_block(contents[block_st:line_id], copy.copy(block_id))
                        block_id += 1
                    block_st = line_id

                line_id += 1

        self.match_between_layers()  # sequence layers

    # match in/out
    def match_between_layers(self):
        prev_in_channel = self.input_size['size'][-1]
        for k in self.route:
            _layer = self.layers[k]
            _layer.num_in = prev_in_channel
            if _layer.num_out == 0:
                _layer.num_out = prev_in_channel
            elif _layer.num_out == -1:  # route layer
                _layer.num_out = 0
                for prevl in range(len(_layer.route_layers)):
                    _jump = _layer.route_layers[prevl]
                    refer_l = self.route[k + _jump if _jump < 0 else _jump]
                    _layer.num_out += self.layers[refer_l].num_out
                    _layer.route_layers[prevl] = self.layers[refer_l].name
            prev_in_channel = _layer.num_out
            print(k, _layer.type, ' | ', _layer.num_in, '->', _layer.num_out)

    # split block lines into dictionary
    def __split_block_opt__(self, block_lines):
        global __empty_info__
        kv_split = list(__empty_info__.sub('', l).split('=') for l in block_lines)
        kv_split = filter(lambda x: len(x) == 2, kv_split)  # filtering invalid line
        return {k[0]: k[1] for k in kv_split}

    def parse_block(self, lines, index):
        header = lines[0]
        # get layer type
        if header == '[net]':
            net_opt = self.__split_block_opt__(lines[1:])
            self.input_size['batch'] = (int)(net_opt['batch'])
            self.input_size['size'] = [
                (int)(net_opt['width']),
                (int)(net_opt['height']),
                (int)(net_opt['channels'])
            ]
        elif header == '[region]':
            pass
        elif header[0] == '[':
            default_name = header[1:-1]
            default_scope = None
            initializer = __parse_layers__[default_name]
            if initializer:
                # use the same keys as cfg file
                layer_opt = self.__split_block_opt__(lines[1:])
                layer_opt['#TYPE'] = default_name
                if '#NAME' not in layer_opt:
                    layer_opt['#NAME'] = str(index)
                if '#SCOPE' not in layer_opt:
                    layer_opt['#SCOPE'] = default_scope
                self.layers[index] = initializer(dtype=self.dtype,
                                                 kwargs=layer_opt)
                self.layers[layer_opt['#NAME']] = self.layers[index]
                self.route.append(index)
            else:
                print('unsupported layer type: %s' % (header))

    def statistcs_size(self, unit='MB', print_out=False, export_csv_as=None):
        total_count = 0.0
        self.statistics['layer_size'] = []  # may to be appled with Pandas in the future

        # index |
        # param count me |
        # param size (:unit) me |
        # param count accumulated |
        # param size (:unit) accumulated |
        # ractions
        for k in self.route:
            _layer = self.layers[k]
            my_size = _layer.my_size(flag='count')
            p_count = sum(my_size.values())
            total_count += p_count
            self.statistics['layer_size'].append([
                str(k),
                _layer.type,
                p_count,
                p_count * __unit_size__[unit] * bwidth.__bwidth__[self.dtype],
                total_count,
                total_count * __unit_size__[unit] * bwidth.__bwidth__[self.dtype],
                None
            ])

        # compute fractions
        for l in self.statistics['layer_size']:
            l[-1] = (float)(l[2]) / (float)(total_count)

        # summary
        self.statistics['summary'] = ([
            total_count,
            total_count * __unit_size__[unit] * bwidth.__bwidth__[self.dtype]
        ])
        """
        table = DF(columns=['index', 'layer type',
                            'param count','param size(%s)'%(unit),
                            'acc. count', 'acc. size(%s)'%(unit),
                            'fraction(%)'],
                   data=self.statistics['layer_size'])

        # formatting
        if print_out or export_csv_as:
            print_table = deepcopy(table)
            print_table['index'] = print_table['index'].map(lambda x:'<%s>'%x)
            print_table['param count'] = print_table['param count'].map(lambda x: '%i'%x)
            print_table['acc. count'] = print_table['acc. count'].map(lambda x: '%i' % x)
            print_table['fraction(%)'] = print_table['fraction(%)'].map(lambda x: '%.2f%%'%(x*100.0))
            self.print_table = print_table

            if print_out:
                print('Data type: ', self.dtype)
                print(print_table)
                print('summary:\ntotal count: %i\ntotal size: %.4f %s'%(
                    self.statistics['summary'][0],
                    self.statistics['summary'][1],
                    unit
                ))

            if export_csv_as is not None:
                print_table.to_csv(export_csv_as,index=False)
        """
