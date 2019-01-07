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

import numpy as np


class RangeFromBatchMinMax:
    def __call__(self, sess, tensor, dataset, is_weights=False):
        batch = sess.run(tensor, dataset)
        minv = batch.min()
        maxv = batch.max()
        print('tensor {} min/max = '.format(tensor.name), minv, maxv)
        return minv, maxv, batch


class RangeFromBatchMinMax98:
    def __call__(self, sess, tensor, dataset, is_weights=False):
        batch = sess.run(tensor, dataset)
        batch_s = sorted(batch.flatten())
        assert (batch.size > 100)
        minv = batch_s[round(len(batch_s) * 0.01)]
        maxv = batch_s[round(len(batch_s) * 0.99)]
        return minv, maxv, batch


class RangeFromBatchMinMax90:
    def __call__(self, sess, tensor, dataset, is_weights=False):
        batch = sess.run(tensor, dataset)
        batch_s = sorted(batch.flatten())
        assert (batch.size > 100)
        minv = batch_s[round(len(batch_s) * 0.05)]
        maxv = batch_s[round(len(batch_s) * 0.95)]
        return minv, maxv, batch


class RangeFromBatchMinMax80:
    def __call__(self, sess, tensor, dataset, is_weights=False):
        batch = sess.run(tensor, dataset)
        batch_s = sorted(batch.flatten())
        assert (batch.size > 100)
        minv = batch_s[round(len(batch_s) * 0.1)]
        maxv = batch_s[round(len(batch_s) * 0.9)]
        return minv, maxv, batch


class RangeFromBatchMeanMinsMaxs:
    def __call__(self, sess, tensor, dataset, is_weights=False):
        if is_weights:
            return RangeFromBatchMinMax()(sess, tensor, dataset, is_weights)
        else:
            batch = sess.run(tensor, dataset)
            n_batch = np.reshape(batch, [batch.shape[0], np.prod(batch.shape[1:])])
            minv = n_batch.min(axis=1).mean()
            maxv = n_batch.max(axis=1).mean()
            return minv, maxv, batch


class RangeFromBatchKL:
    BINS_NUMBER = 8192
    QUANTIZE_SIZE = 256

    def chunks(self, l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]

    def smooth(self, y, box_pts):
        box = np.ones(box_pts) / box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth

    def quantize_x(self, origin, x):
        chunked_data = list(self.chunks(origin, len(origin) // x))

        foo = [sum(i) for i in chunked_data]
        final_array = []

        for m, piece in enumerate(chunked_data):

            weight = foo[m]
            if weight == 0:
                final_array += [0] * len(piece)
                continue

            binary_piece = np.array(piece > 0)
            replace_val = foo[m] / sum(binary_piece)
            final_array += list(replace_val * binary_piece)

        return final_array

    def calc_kld(self, P, start_bin_max, end_bin_max, start_bin_min, end_bin_min, delta, max_val, min_val):
        import scipy.stats
        from copy import deepcopy
        klds = {}
        for i in range(start_bin_max, end_bin_max + 1, self.QUANTIZE_SIZE):
            for j in range(start_bin_min, end_bin_min + 1, self.QUANTIZE_SIZE):
                reference_distribution_P = deepcopy(P[j:i])
                left_outliers_count = np.sum(P[0:j])
                right_outliers_count = np.sum(P[i:self.BINS_NUMBER])

                reference_distribution_P[0] += left_outliers_count
                reference_distribution_P[-1] += right_outliers_count

                candidate_distribution_Q = self.quantize_x(reference_distribution_P, self.QUANTIZE_SIZE)
                left_outliers_P = deepcopy(P[:j + (i - j) // self.QUANTIZE_SIZE])
                right_outliers_P = deepcopy(P[i - (i - j) // self.QUANTIZE_SIZE:])
                left_replace_val = 0
                if sum(left_outliers_P > 0) > 0:
                    left_replace_val = sum(left_outliers_P) / sum(left_outliers_P > 0)
                right_replace_val = 0
                if sum(right_outliers_P > 0) > 0:
                    right_replace_val = sum(right_outliers_P) / sum(right_outliers_P > 0)
                candidate_distribution_Q = list(left_replace_val * (left_outliers_P > 0)) + candidate_distribution_Q[(
                                                                                                                                 i - j) // self.QUANTIZE_SIZE:i - j - (
                            i - j) // self.QUANTIZE_SIZE] + list(right_replace_val * (right_outliers_P > 0))

                Q = np.array(candidate_distribution_Q)

                kld = scipy.stats.entropy(P, Q)

                # print((j,i), kld, (j + 0.5) * delta + (min_val - delta), (i + 0.5) * delta + (min_val - delta))
                klds[(j, i)] = kld

        return klds

    def convert_layer_output(self, data):
        image_num = data.shape[0]

        max_all = np.max(data)
        min_all = np.min(data)
        delta = (max_all - min_all) / (self.BINS_NUMBER + 1)
        bins_all = np.arange(min_all, max_all, delta)  # fixed bin size

        P = np.zeros(self.BINS_NUMBER)
        for image_idx in range(image_num):
            data_curr_image = np.ndarray.flatten(data[image_idx])

            n, bins = np.histogram(data, bins=bins_all)
            P = P + n

        return (P, min_all, max_all, delta)

    def find_min_max_kld(self, data):
        (P, min_data, max_data, delta) = self.convert_layer_output(data)
        P = self.smooth(P, 512)
        # find max first
        klds_max = self.calc_kld(P, self.QUANTIZE_SIZE, self.BINS_NUMBER, 0, 0, delta, max_data, min_data)
        (tmp, max_bin) = min(zip(klds_max.values(), klds_max.keys()))[1]
        klds_min = self.calc_kld(P, max_bin, max_bin, 0, max_bin - 1, delta, max_data, min_data)
        (min_bin, tmp) = min(zip(klds_min.values(), klds_min.keys()))[1]

        threshold_min = (min_bin) * delta + (min_data)
        threshold_max = (max_bin) * delta + (min_data)
        print('Min data', 'idx', threshold_min)
        print('Max data', 'idx', threshold_max)

        return (threshold_min, threshold_max)

    def __call__(self, sess, tensor, dataset, is_weights=False):
        if is_weights:
            return RangeFromBatchMinMax()(sess, tensor, dataset, is_weights)
        else:
            batch = sess.run(tensor, dataset)
            minv, maxv = self.find_min_max_kld(batch)
            return minv, maxv, batch
