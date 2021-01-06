
## Dependent
you need `python3` `tensorflow` and `pillow`.
```sh
pip3 install tensorflow
pip3 install pillow
```

## install
`git clone <this model-compiler>`

## plugins
### dataset loader
`dataset_loader/img_0_1.py` loads image and convert it as `0`~`1` range.
+ required options: `--dataset_pic_path` `--image_w` `--image_h`

`dataset_loader/img_neg1_1.py` loads image and convert it as `-1`~`1` range.
+ required options: `--dataset_pic_path` `--image_w` `--image_h`

### model loader
`model_loader/pb` loads TensorFlow model.
+ required options: `--pb_path` `--dataset_input_name` `--tensor_output_name`
+ optional options: `--tensorboard_mode` `--tensor_input_name`
`--tensor_input_min` `--tensor_input_max` `--tensor_input_minmax_auto` 
`--eight_bit_mode` `--layer_start_idx`

`model_loader/h5` loads Keras model.
+ required options: `--h5_path` `--dataset_input_name` `--tensor_output_name`
+ optional options: `--tensorboard_mode` `--tensor_input_name`
`--tensor_input_min` `--tensor_input_max` `--tensor_input_minmax_auto` 
`--eight_bit_mode` `--layer_start_idx`

`model_loader/darknet` loads DarkNet model.
you should add `pad=0` in pooling section in config file.
+ required options: `--cfg_path` `--weights_path`
+ optional options: `--tensorboard_mode`
`--tensor_input_min` `--tensor_input_max` `--tensor_input_minmax_auto` 
`--eight_bit_mode` `--layer_start_idx`


## usage
`python3 model-compiler --dataset_loader <dataset_loader_path>
 --model_loader <model_loader_path> <options>`
 
 required options:
 `--dataset_input_name`
 
 optional options:
 `--output_path` `--eight_bit_mode` `--prefix` `--layer_start_idx`

## example
```sh
cd kendryte-model-compiler
python3 . --dataset_input_name input:0 \
 --dataset_loader dataset_loader/img_0_1.py \
 --image_h 240 --image_w 320 \
 --dataset_pic_path dataset/yolo_240_320 \
 --model_loader model_loader/pb \
 --pb_path pb_files/20classes_yolo.pb --tensor_output_name yv2
```

## Q&A
Q: how to show more help?\
A: use `-h` option.

Q: what is `--tensorboard_mode`? \
A: show your model in tensorboard(https://www.tensorflow.org/guide/summaries_and_tensorboard).

Q: what is `--eight_bit_mode`?\
A: your weights stores in `16bit` mode by default, this option let your weithts stores in `8bit` mode.
