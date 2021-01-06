from . import format_tool

# supported Darknet layer types
# : expanded name abrev.
__layer_types__ = {
    'depthwise_convolutional': 'dw_conv',
    'convolutional': 'conv',
    'connected': 'fc',
    'route_concat': 'concat',
    'route_sum': 'residue',
    'maxpool': 'maxpool',
    'avgpool': 'avgpool',
    'softmax': 'softmax'
}
# supported activation functions
__activation_fn__ = {
    'leaky': 'leaky_relu',
    'relu': 'tf.nn.relu',
    'tanh': 'tf.nn.tanh',
    'logistic': 'tf.nn.sigmoid',
    'linear': 'None'
}

# data unit width
__size_flag__ = {
    'count': 1,
    'byte': 1,
    'float32': 4
}


# basic parent class
class _basic_layer_(object):
    def __init__(self, name, layer_type, scope=None, dtype='float32'):
        self.scope = scope
        # check if the layer type is supported
        # print(layer_type)
        assert layer_type in __layer_types__, 'unsupported layer type: %s' % layer_type
        self.type = layer_type
        self.name = '{}_{}'.format(name, __layer_types__[layer_type]) \
            if name.isdigit() else name

        self.num_in = 0
        self.num_out = 0
        self.tf_out = 0

        self.kernel_size = 0

        self.reuse = False
        self.trainable = False

    def __str__(self):
        return 'Name: %s\nScope: %s\nLayer type: %s' % (
            self.name,
            './' if self.scope is None else self.scope,
            self.type
        )


# convolutional layer
class convolutional_layer(_basic_layer_):
    def __init__(self, dtype, kwargs):
        super(convolutional_layer, self).__init__(kwargs['#NAME'],
                                                  'convolutional',
                                                  kwargs['#SCOPE'],
                                                  dtype=dtype)
        self.num_out = (int)(kwargs['filters'])
        self.tf_out = self.num_out
        self.kernel_size = (int)(kwargs['size'])
        self.stride = (int)(kwargs['stride'])
        self.padding = 'SAME'
        if 'pad' in kwargs:
            self.padding = 'VALID' if kwargs['pad'] == '0' else 'SAME'
        self.activation_fn = kwargs['activation']

        self.batch_normalize = False
        if 'batch_normalize' in kwargs:
            self.batch_normalize = (bool)((int)(kwargs['batch_normalize']))

    def __str__(self):
        # basic info
        # in->out | ksize x ksize / stride |
        return super(convolutional_layer, self).__str__() + \
               '\n%i -> %i \t %ix%i/%i\n' % (self.num_in, self.num_out,
                                             self.kernel_size, self.kernel_size, self.stride) + \
               'padding: %s\n' % (self.padding) + \
               'batch_normalize: %s\n' % (str(self.batch_normalize)) + \
               'activation: %s' % (self.activation_fn)

    def my_size(self, flag='byte'):  # size flag
        cross_num = self.num_in * self.num_out
        bw = __size_flag__[flag]
        b_size = self.num_out * bw
        return {
            'weights': cross_num * self.kernel_size * self.kernel_size * bw,
            'bias': b_size,
            'bn_scale': b_size if self.batch_normalize else 0,
            'bn_mean': b_size if self.batch_normalize else 0,
            'bn_variance': b_size if self.batch_normalize else 0
        }

    def to_tf_code(self, template, load_darknet=False, indent=0, indent_unit='\t'):
        code = template
        # remove the bn args
        if not self.batch_normalize:
            bn_interval = [code.find('bn_scale'), code.find('parent_scope')]
            code = code[:bn_interval[0]] + code[bn_interval[1]:]
        # formatting
        if load_darknet:  # load class >darknet
            code = code.format([
                self.name,
                self.stride, self.padding,
                __activation_fn__[self.activation_fn],
                str(self.batch_normalize),
                self.scope if self.scope else 'None',
                self.name,
                str(self.reuse)],
                format_tool.indent_chars(indent, indent_unit)
            )
        else:
            pass

        return code


# depthwise-convolutional layer
class dw_convolutional_layer(_basic_layer_):
    def __init__(self, dtype, kwargs):
        super(dw_convolutional_layer, self).__init__(kwargs['#NAME'],
                                                     'depthwise_convolutional',
                                                     kwargs['#SCOPE'],
                                                     dtype=dtype)
        self.kernel_size = (int)(kwargs['size'])
        self.tf_out = 1
        self.stride = (int)(kwargs['stride'])
        self.padding = 'SAME'
        if 'pad' in kwargs:
            self.padding = 'VALID' if kwargs['pad'] == '0' else 'SAME'
        self.activation_fn = kwargs['activation']

        self.batch_normalize = False
        if 'batch_normalize' in kwargs:
            self.batch_normalize = (bool)((int)(kwargs['batch_normalize']))

    def convert_to_tf_code(self, indent=0):
        pass

    def __str__(self):
        # basic info
        # in->out | ksize x ksize / stride |
        return super(dw_convolutional_layer, self).__str__() + \
               '\n%i -> %i \t %ix%i/%i\n' % (self.num_in, self.num_out,
                                             self.kernel_size, self.kernel_size, self.stride) + \
               'padding: %s\n' % self.padding + \
               'batch_normalize: %s\n' % (str(self.batch_normalize)) + \
               'activation: %s' % self.activation_fn

    def my_size(self, flag='byte'):  # size flag
        bw = __size_flag__[flag]
        b_size = self.num_in * bw
        return {
            'weights': b_size * self.kernel_size * self.kernel_size,
            'bias': b_size,
            'bn_scale': b_size if self.batch_normalize else 0,
            'bn_mean': b_size if self.batch_normalize else 0,
            'bn_variance': b_size if self.batch_normalize else 0
        }

    def to_tf_code(self, template, load_darknet=False, indent=0, indent_unit='\t'):
        code = template
        # remove the bn args
        if not self.batch_normalize:
            bn_interval = [code.find('bn_scale'), code.find('parent_scope')]
            code = '{}{}'.format(code[:bn_interval[0]], code[bn_interval[1]:])
        # formatting
        if load_darknet:  # load class >darknet
            code = code.format([
                self.name,
                self.stride, self.padding,
                __activation_fn__[self.activation_fn],
                str(self.batch_normalize),
                self.scope if self.scope else 'None',
                self.name,
                str(self.reuse)],
                format_tool.indent_chars(indent, indent_unit)
            )
        else:
            pass

        return code


# maxpool layer
class maxpooling_layer(_basic_layer_):
    def __init__(self, dtype, kwargs):
        super(maxpooling_layer, self).__init__(kwargs['#NAME'],
                                               'maxpool',
                                               kwargs['#SCOPE'],
                                               dtype=dtype)
        self.kernel_size = (int)(kwargs['size']) if 'size' in kwargs else 0  # if not specified => reduce_max
        self.stride = (int)(kwargs['stride']) if 'stride' in kwargs else 0

        self.padding = 'SAME'
        if 'pad' in kwargs:
            self.padding = 'VALID' if kwargs['pad'] == '0' else 'SAME'

        self.activation_fn = 'linear'
        if 'activation' in kwargs:
            self.activation_fn = kwargs['activation']

        self.batch_normalize = False
        if 'batch_normalize' in kwargs:
            self.batch_normalize = (bool)((int)(kwargs['batch_normalize']))

    def convert_to_tf_code_slim(self, indent=0):
        pass

    def __str__(self):
        # basic info
        # in->out | ksize x ksize / stride |
        return super(maxpooling_layer, self).__str__() + \
               '\n%i -> %i \t %ix%i/%i\n' % (self.num_in, self.num_out,
                                             self.kernel_size, self.kernel_size, self.stride) + \
               'padding: %s\n' % (self.padding) + \
               'batch_normalize: %s\n' % (str(self.batch_normalize)) + \
               'activation: %s' % (self.activation_fn)

    def my_size(self, flag='byte'):  # size flag
        return {'weights': 0}

    def to_tf_code(self, template, load_darknet=False, indent=0, indent_unit='\t'):
        return template.format(
            [self.kernel_size, self.stride,
             self.padding, self.scope if self.scope else 'None', self.name],
            format_tool.indent_chars(indent, indent_unit)
        )


# avgpool layer
class avgpooling_layer(_basic_layer_):
    def __init__(self, dtype, kwargs):
        super(avgpooling_layer, self).__init__(kwargs['#NAME'],
                                               'avgpool',
                                               kwargs['#SCOPE'],
                                               dtype=dtype)
        self.kernel_size = (int)(kwargs['size']) if 'size' in kwargs else 0  # if not specified => reduce_mean
        self.stride = (int)(kwargs['stride']) if 'stride' in kwargs else 0

        self.padding = 'SAME'
        if 'pad' in kwargs:
            self.padding = 'VALID' if kwargs['pad'] == '0' else 'SAME'

        self.activation_fn = 'linear'
        if 'activation' in kwargs:
            self.activation_fn = kwargs['activation']

        self.batch_normalize = False
        if 'batch_normalize' in kwargs:
            self.batch_normalize = (bool)((int)(kwargs['batch_normalize']))

    def convert_to_tf_code_slim(self, indent=0):
        pass

    def __str__(self):
        # basic info
        # in->out | ksize x ksize / stride |
        return super(avgpooling_layer, self).__str__() + \
               '\n%i -> %i \t %ix%i/%i\n' % (self.num_in, self.num_out,
                                             self.kernel_size, self.kernel_size, self.stride) + \
               'padding: %s\n' % (self.padding) + \
               'batch_normalize: %s\n' % (str(self.batch_normalize)) + \
               'activation: %s' % (self.activation_fn)

    def my_size(self, flag='byte'):  # size flag
        return {'weights': 0}

    def to_tf_code(self, template, load_darknet=False, indent=0, indent_unit='\t'):
        return template.format(
            [self.kernel_size, self.stride,
             self.padding, self.scope if self.scope else 'None', self.name],
            format_tool.indent_chars(indent, indent_unit)
        )


# route/shortcut/residue layer
__route_merge_method__ = {
    'route': 'route_concat',
    'shortcut': 'route_sum',
}


class route_layer(_basic_layer_):
    def __init__(self, dtype, kwargs):
        super(route_layer, self).__init__(kwargs['#NAME'],
                                          __route_merge_method__[kwargs['#TYPE']],
                                          kwargs['#SCOPE'],
                                          dtype=dtype)
        self.activation_fn = 'linear'
        if 'activation' in kwargs:
            self.activation_fn = kwargs['activation']

        route_key = None
        self.route_layers = [-1]
        if 'layers' in kwargs:
            route_key = 'layers'
        elif 'from' in kwargs:
            route_key = 'from'

        if route_key:
            raw_members = kwargs[route_key].split(',')
            self.route_layers = list((int)(_l.strip()) for _l in raw_members)
            self.num_out = -1  # set num_out when forward flow

    def __str__(self):
        # basic info
        # in->out | ksize x ksize / stride |
        return super(route_layer, self).__str__() + \
               'merged by: %s' % (self.type) + \
               '\nlayers route: [%s]' % (
                   (''.join('end_node[%i],' % (s) for s in self.route_layers))[:-1])

    def my_size(self, flag='byte'):  # size flag
        return {'weights': 0}

    def to_tf_code(self, template, load_darknet=False, indent=0, indent_unit='\t'):
        # route to string
        if len(self.route_layers) == 0:
            return '#INVALID ROUTE LAYER at layer <{}>\n'.format(self.name)
        route_str = (''.join('end_node[\'%s\'],' % (s) for s in self.route_layers))[:-1]

        # list of formatted entries
        format_list = [route_str]  # route indices
        # specific methods
        # -------------------------- #
        if self.type != 'route_concat':
            format_list.append(__activation_fn__[self.activation_fn])
        # -------------------------- #
        format_list.append(self.name)  # name

        return template.format(format_list, format_tool.indent_chars(indent, indent_unit))
