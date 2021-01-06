import os
import re
import shutil
from .format_tool import indent_chars

__class_indent__ = {
    'global': '',
    'output': '{0}'
}

# laod code pattern
__py_tf_code__ = dict()
current_dir = os.path.dirname(__file__)
__template__ = os.path.join(current_dir, 'static_lib/tf_template')
with open(__template__, 'r') as _f_code:
    contents = _f_code.readlines()
    _class = None
    _label = None
    for l in contents:
        # get code block
        if l[0] == '>':
            _class, _label = l[1:].strip().split(',')
            if _class not in __py_tf_code__:
                __py_tf_code__[_class] = {}
            __py_tf_code__[_class][_label] = ''
        elif _class:
            # for indentent
            __indent__ = __class_indent__[_class] \
                if _class in __class_indent__ \
                else '{1}'
            __py_tf_code__[_class][_label] += (__indent__ + l)


def compile_into_TFW(darknet, out_tfw_path, dtype=None):
    F_tfw = open(out_tfw_path, 'wb')
    for i in darknet.net.route:
        _l = darknet.net.layers[i]
        # reshape as tf format
        darknet.decode_buf(i,
                           ((_l.tf_out, _l.num_in, _l.kernel_size, _l.kernel_size),
                            (2, 3, 1, 0)))
        # encode into .TFW file (tensorflow weights)
        enc_arr = darknet.encode_buf(_l.name, dtype)
        for arr in enc_arr:
            F_tfw.write(arr)

    F_tfw.close()


__d2t_static_list__ = [

]


def darknet_to_tf_module(darknet, reuse=None, out_dir=None, new_dtype=None):
    # check the existence of out_dir
    d2t_dir = os.path.join(out_dir, 'd2t')
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    if not os.path.isdir(d2t_dir):
        shutil.copytree(os.path.join(current_dir, 'static_lib'), d2t_dir)

    # copy the static_lib

    # compile darknet_network
    # storing weights into .tfw
    if darknet.__own_weights__:
        compile_into_TFW(darknet, os.path.join(out_dir, darknet.name + '.tfw'),
                         dtype=new_dtype)

    # complie darknet_network structure into tensorflow python code
    with open(os.path.join(out_dir, '%s.py' % (darknet.name)), 'w') as FW:
        FW.write(__py_tf_code__['global']['header'])
        # data type
        FW.write(__py_tf_code__['global']['data_type'].format(darknet.dtype))

        # load TFW
        FW.write(__py_tf_code__['darknet']['read_tfw'].format(
            '%s.tfw' % (darknet.name), ''))

        load_at = 0
        load_code = []  # for loading weights
        # make_code = []  # for making variables
        op_code = []  # for operations
        for i in darknet.net.route:
            _l = darknet.net.layers[i]

            # operation
            template = __py_tf_code__['darknet'][_l.type]
            op_code.append(_l.to_tf_code(template, True, 1, '\t'))

            # loading
            l_size = _l.my_size(darknet.dtype)  # x data width
            jump_dist = sum(l_size.values())
            if jump_dist == 0:
                continue
            # b->s->m->v->w
            jump_dist = l_size['bias']
            _data_indent = 2
            load_code.append(__py_tf_code__['darknet']['load_b'].format(
                [_l.name, load_at, load_at + l_size['bias'], str(_l.trainable)],
                indent_chars(_data_indent)
            ))
            load_at += jump_dist

            if _l.batch_normalize:
                jump_dist = l_size['bn_scale']
                for _c_ in ['s', 'm', 'v']:
                    load_code.append(__py_tf_code__['darknet']['load_' + _c_].format(
                        [_l.name, load_at, load_at + l_size['bn_scale'], str(_l.trainable)],
                        indent_chars(_data_indent)
                    ))
                    load_at += jump_dist

            jump_dist = l_size['weights']
            load_code.append(__py_tf_code__['darknet']['load_w'].format(
                [_l.name, load_at, load_at + l_size['weights'],
                 '[{0},{0},{1},{2}]'.format(_l.kernel_size, _l.num_in, _l.tf_out),
                 str(_l.trainable)],
                indent_chars(_data_indent)
            ))
            load_at += jump_dist

        # LOAD DATA function
        var_scope = '%s_var' % (darknet.name)
        FW.write(__py_tf_code__['global']['def_loaddata'])
        FW.write(__py_tf_code__['scope']['var_scope'].format(
            [darknet.name, var_scope], indent_chars(1)))
        FW.write(''.join(c for c in load_code))
        # NN function
        FW.write(__py_tf_code__['global']['def_net'])
        FW.write(''.join(c for c in op_code))

        # return
        FW.write(__py_tf_code__['output']['net_out'].format(indent_chars(1)))

    ## bwidth.py
    with open(os.path.join(out_dir, 'bwidth.py'), 'w') as F_init:
        F_init.write(__py_tf_code__['global']['__init__'].format(
            re.split(r'[\\/]', out_dir)[-1], darknet.name
        ))

    print('debug:', os.path.join(out_dir, 'info.txt'))
    # extra info
    with open(os.path.join(out_dir, 'info.txt'), 'w') as F_info:
        F_info.write("#input info\n")
        F_info.write("data type: %s\n" % (darknet.dtype))
        print(darknet.net.input_size['size'])
        F_info.write("width: {0[0]}\nheight: {0[1]}\nchannel: {0[2]}\n".format(
            darknet.net.input_size['size']))
