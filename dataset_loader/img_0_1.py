import os
import numpy as np
from PIL import Image


def box_image(im_path, new_w, new_h):
    orig = Image.open(im_path)
    w, h = orig.size
    w_scale = float(new_w) / w
    h_scale = float(new_h) / h

    n_w = new_w
    n_h = new_h
    if w_scale < h_scale:
        n_h = int(h * w_scale)
    else:
        n_w = int(w * h_scale)

    ch_size = {'RGB': 3}.get(orig.mode, 1)
    resized = np.array(orig.resize([n_w, n_h]), dtype='float32') / 255.0
    resized = resized.reshape([*resized.shape, ch_size][:3])

    box_im = np.ones([new_h, new_w, ch_size], dtype='float32') * 0.5
    fill_y = (new_h - n_h) >> 1
    fill_x = (new_w - n_w) >> 1
    box_im[fill_y:fill_y + n_h, fill_x:fill_x + n_w, :] = resized

    return box_im, resized


def load_dataset(args):
    if os.path.isdir(args.dataset_pic_path):
        import random
        all_files = os.listdir(args.dataset_pic_path)
        if len(all_files) > 128:
            print(
                '[warning] you have too many dataset, may slow down this process. '
                'force sampled to 128 items of them.'
            )
            all_files = random.sample(all_files, 128)  # set maxmum dataset size

        dataset_file_list = [
            os.path.join(args.dataset_pic_path, f)
            for f in all_files
            if os.path.isfile(os.path.join(args.dataset_pic_path, f))
        ]
    else:
        dataset_file_list = (args.dataset_pic_path,)

    dataset_val = np.array([box_image(path, args.image_w, args.image_h)[0].tolist() for path in dataset_file_list])
    return dataset_val
