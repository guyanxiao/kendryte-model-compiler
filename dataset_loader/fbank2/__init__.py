import os

import numpy as np


def load_file(filename, file_format, frame_rate=16000):
    import pydub
    from pydub import effects
    sound = pydub.AudioSegment.from_file(filename, file_format)
    sound = sound.set_frame_rate(frame_rate)
    sound = sound.set_channels(1)
    sound = sound.set_sample_width(2)
    sound = sound.remove_dc_offset()
    sound = effects.normalize(sound)
    return np.array(sound.get_array_of_samples())


def fft_singal(singal, pre_frame, window_size=512, shift_size=160, window_func=(lambda x: np.ones((x,))), nfft=512):
    import python_speech_features
    singal = pre_frame(singal) if pre_frame is not None else singal
    frames = python_speech_features.sigproc.framesig(singal, window_size, shift_size, window_func)
    complex_spec = np.fft.rfft(frames, nfft)
    return complex_spec.astype('complex64')


def fbank_from_complex_spec(complex_spec, nfilt=64, nfft=512, sample_rate=16000):
    import python_speech_features
    power = 1 / nfft * np.square(complex_spec).real
    fb = python_speech_features.get_filterbanks(nfilt, nfft, sample_rate)
    feat = np.dot(power, fb.T)
    feat = np.where(feat == 0, np.finfo(float).eps, feat)
    return feat.astype('float32')


def dleta_fbank(feat):
    last = np.zeros(feat[0].shape)
    ret = np.zeros(feat.shape)
    for item, idx in zip(feat, range(feat.shape[0])):
        dleta = item - last
        ret[idx, :] = dleta
    return ret.astype('float32')


def process_data_single(filename):
    try:
        signal = load_file(filename, 'wav')
        complex_spec = fft_singal(signal, None)
        fbank = fbank_from_complex_spec(complex_spec, 64, 512)
        dleta1 = dleta_fbank(fbank)
        dleta2 = dleta_fbank(dleta1)
        return [filename, complex_spec, fbank, dleta1, dleta2]
    except Exception as e:
        print('[error]', filename, e)
        return None


def test_file(filename):
    [filename, complex_spec, fbank, dleta1, dleta2] = process_data_single(filename)
    input_feature = np.zeros((64, 256, 1))
    if fbank.shape[0] <= 256:
        input_feature[:, 0:fbank.shape[0], 0] = fbank.transpose()
    else:
        input_feature[:, :, 0] = fbank[0:256, :].transpose()
    input_feature = np.clip(input_feature, 0.0001, None)
    input_feature = np.log(input_feature)
    input_feature = (input_feature - np.average(input_feature)) / np.std(input_feature)
    input_feature = np.transpose(input_feature, (2, 0, 1))
    return input_feature


def load_dataset(args):
    # feature_extraction = tools.import_from_path(os.path.join(os.path.dirname(__file__), 'feature_extraction.py'))
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

    fbank_list = []
    for filename in dataset_file_list:
        fbank = np.array(test_file(filename))
        fbank_list.append(fbank)

    return np.array(fbank_list)
