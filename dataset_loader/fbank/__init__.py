import os
import numpy as np
import tools


def load_dataset(args):
    #feature_extraction = tools.import_from_path(os.path.join(os.path.dirname(__file__), 'feature_extraction.py'))
    from . import feature_extraction
    if os.path.isdir(args.dataset_path):
        import random
        all_files = os.listdir(args.dataset_path)
        if len(all_files) > 128:
            print(
                '[warning] you have too many dataset, may slow down this process. '
                'force sampled to 128 items of them.'
            )
            all_files = random.sample(all_files, 128)  # set maxmum dataset size

        dataset_file_list = [
            os.path.join(args.dataset_path, f)
            for f in all_files
            if os.path.isfile(os.path.join(args.dataset_path, f))
        ]
    else:
        dataset_file_list = (args.dataset_path,)

    max_pad_len = 256
    fbank_list = []
    for filename in dataset_file_list:
        fbank = np.array(feature_extraction.wav2fbank(filename, max_pad_len))
        fbank = fbank.reshape(64, max_pad_len, 1)
        fbank_list.append(fbank)

    return np.array(fbank_list).transpose([0, 3, 1, 2])
