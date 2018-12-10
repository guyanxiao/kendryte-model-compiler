import tempfile

import tensorflow as tf
from tensorflow.python.platform import gfile

import range_from_batch
import tools

from . import tensor_list_to_layer_list
from . import tensor_head_to_tensor_list
from . import layer_list_to_k210_layer


def load_graph(pb_file_path, tensor_output_name, tensor_input_name):
    if pb_file_path.endswith('h5'):
        import h5_converter
        pb_file_path = h5_converter.convert(pb_file_path)

    if pb_file_path.endswith('pb'):
        with tf.Session() as persisted_sess:
            with gfile.GFile(pb_file_path, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                persisted_sess.graph.as_default()
                tf.import_graph_def(graph_def, name='')

        output_tensor, input_tensor = None, None
        if tensor_output_name is not None:
            output_tensor = persisted_sess.graph._nodes_by_name[tensor_output_name].outputs[0]
        if tensor_input_name is not None:
            input_tensor = persisted_sess.graph._nodes_by_name[tensor_input_name].outputs[0]

        return output_tensor, input_tensor

    return None


def load_model(dataset, args):
    if args.tensorboard_mode:
        load_graph(args.pb_path, None, None)
        graphs_path = tempfile.mkdtemp('graphs')
        writer = tf.summary.FileWriter(graphs_path, tf.Session().graph)
        writer.close()
        import subprocess
        subprocess.call(['tensorboard', '--logdir', graphs_path])
        exit(0)


    tensor_output, tensor_input = load_graph(args.pb_path, args.tensor_output_name, args.tensor_input_name)
    with tf.Session() as sess:
        converter = tensor_head_to_tensor_list.PbConverter(tensor_output, tensor_input)
        converter.convert()
        layers = tensor_list_to_layer_list.convert_to_layers(sess, dataset, converter.dst)

        rfb = range_from_batch.RangeFromBatchMinMax()
        if args.tensor_input_minmax_auto:
            input_min, input_max, = rfb(sess, tensor_input, dataset)
            in_scale, in_bias = tools.min_max_to_scale_bias(input_min, input_max)
            print('[layer input] scale/bias:', in_scale, in_bias)

        k210_layers = layer_list_to_k210_layer.gen_k210_layers(
            layers, sess, dataset,
            range_from_batch=rfb,
            eight_bit_mode=args.eight_bit_mode,
            input_min=args.tensor_input_min,
            input_max=args.tensor_input_max,
            layer_start_idx=args.layer_start_idx
        )

    return k210_layers
