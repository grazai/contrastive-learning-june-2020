import numpy as np
import random
import operator
import functools
import itertools

# TODO: I add some ideas for augmentation from other ressources
# TODO: adding drift
# ( see https://tsaug.readthedocs.io/en/stable/notebook/Examples%20of%20augmenters.html )
# TODO: add quantisize
# ( see https://tsaug.readthedocs.io/en/stable/notebook/Examples%20of%20augmenters.html )
# TODO: adding scaling
# ( see https://github.com/terryum/Data-Augmentation-For-Wearable-Sensor-Data/blob/master/Example_DataAugmentation_TimeseriesData.ipynb )
# TODO: adding magnitute warping
# ( see https://github.com/terryum/Data-Augmentation-For-Wearable-Sensor-Data/blob/master/Example_DataAugmentation_TimeseriesData.ipynb )
# TODO: adding Time Warping
# ( see https://github.com/terryum/Data-Augmentation-For-Wearable-Sensor-Data/blob/master/Example_DataAugmentation_TimeseriesData.ipynb )
# ( see https://tsaug.readthedocs.io/en/stable/notebook/Examples%20of%20augmenters.html )
# ( also in paper: https://halshs.archives-ouvertes.fr/halshs-01357973/document )
# TODO: adding rotation / reverse
# ( see https://github.com/terryum/Data-Augmentation-For-Wearable-Sensor-Data/blob/master/Example_DataAugmentation_TimeseriesData.ipynb )
# ( see https://tsaug.readthedocs.io/en/stable/notebook/Examples%20of%20augmenters.html )
# TODO: adding permutation
# ( see https://github.com/terryum/Data-Augmentation-For-Wearable-Sensor-Data/blob/master/Example_DataAugmentation_TimeseriesData.ipynb )
# TODO: add random samplint
# ( see https://github.com/terryum/Data-Augmentation-For-Wearable-Sensor-Data/blob/master/Example_DataAugmentation_TimeseriesData.ipynb )
# TODO: add DBS algorithm
# ( see https://github.com/hfawaz/aaltd18 )
# ( paper: https://germain-forestier.info/publis/aaltd2018.pdf )
# TODO: add random slicing (small offset jittering)
# ( paper: https://halshs.archives-ouvertes.fr/halshs-01357973/document )
# TODO: add crop aka: taking a smaller part of the signal and expand it to original length
# ( see https://tsaug.readthedocs.io/en/stable/notebook/Examples%20of%20augmenters.html )
# TODO: add convolving with a scipy kernel
# ( like in https://tsaug.readthedocs.io/en/stable/notebook/Examples%20of%20augmenters.html )
# TODO: use statistical residuals from this paper to generate new similar signals
# ( paper: https://www.researchgate.net/publication/291421520_Bagging_exponential_smoothing_methods_using_STL_decomposition_and_Box-Cox_transformation )

# TODO: look into this paper to see if you find anything extra interesting there:
# ( paper: https://arxiv.org/pdf/2002.12478.pdf )


def random_splits(indices, test_size, valid_size):
    n = len(indices)
    np.random.shuffle(indices)
    split = int(np.floor(test_size * n))
    train_and_valid_idx, test_idx = indices[split:], indices[:split]
    n_tv = len(train_and_valid_idx)
    split = int(np.floor(valid_size * n_tv))
    train_idx, valid_idx = train_and_valid_idx[split:], train_and_valid_idx[:split]

    return train_idx, valid_idx, test_idx


def add_mark_missing_column(data):
    zrs = np.zeros((data.shape[0], data.shape[1] + 1))
    zrs[:, :-1] = data
    return zrs


def basic_representation(data, num_rows_per_second, desired_rows_per_second, time_axis=0, metadata_axis=1):
    """
    `basic_representation` takes in a 2D dataframe and expands the dataframe from a given number of records per second
    do a desired number of records per second. It returns a 2D dataframe where the time dimension is always along the
    rows
    """
    assert num_rows_per_second < desired_rows_per_second, "we assume we approximate to a maximal frequency given the data here"
    assert len(data.shape) == 2, "we currently only consider 2D datasets"

    n, m = data.shape[time_axis], data.shape[metadata_axis]
    idx = np.array(list(range(n))) * (desired_rows_per_second/num_rows_per_second)
    idx = idx.astype(np.int)
    nn = int((n/num_rows_per_second)*desired_rows_per_second)
    new_shape = (nn, m + 1)
    result = np.zeros(new_shape)
    result[idx, :-1] = data if time_axis==0 else np.transpose(data)
    neg_idx = [i for i in list(range(nn)) if i not in idx]
    result[neg_idx, -1] = 1
    return result


def get_data_idx(data, debug_missing_vals_column=True):
    assert len(data.shape) == 2
    assert data.shape[1] > 1
    if data.shape[1] == 2:
        if debug_missing_vals_column: print('data with only two columns is assumed to have no missing value field, just return the whole index')
        return list(range(len(data)))
    data_idx = np.argwhere(data[:, -1] == 0).squeeze().tolist()
    return data_idx


def downsample(data, remove_every, is_all_data):
    data = np.copy(data)
    if data.shape[1] == 2: # no 'missing data column yet, add it
        data = add_mark_missing_column(data)
    data_idx = get_data_idx(data) if not is_all_data else list(range(len(data)))
    remove_idx = []
    cnt = 0
    for i in range(len(data_idx)):
        if cnt % remove_every == 0:
            remove_idx.append(data_idx[i])
        cnt += 1
    data[remove_idx, 1] = 0.
    data[remove_idx, 2] = 1
    return data


def random_segments(ts_length, seg_length_multiple):
    n = ts_length // seg_length_multiple
    p = 0
    for _ in range(n - 1):
        p += 1
        if random.randrange(2):
            yield p * seg_length_multiple
            p = 0
    yield (p + 1) * seg_length_multiple


def create_time_series_segment(segment_length, segment_offset, ts_types, ts_generator):
    type_idx = int(np.random.uniform(0, len(ts_types)))
    choosen_type = ts_types[type_idx]
    generator = ts_generator[choosen_type]
    segment = list(generator(segment_length, segment_offset))
    return segment, choosen_type


def rem_last_type(ts_types, last_type):
    if last_type is None:
        return ts_types.copy()
    t = ts_types.copy()
    t.remove(last_type)
    return t


def sinus(x, offset, points_per_pi):
    times_pi = x / points_per_pi
    time_steps = np.linspace(0, times_pi * np.pi, x)
    time_steps = time_steps + offset
    data = np.sin(time_steps)
    for ts, s in zip(time_steps, data): yield (ts, s)


def square(x, offset, points_per_square, square_jiggle_offsets):
    times_pi = x / points_per_square
    time_steps = np.linspace(0, times_pi * np.pi, x)
    time_steps = time_steps + offset
    val = 1
    cnt = 0
    interval = points_per_square + np.random.random_integers(0, square_jiggle_offsets)
    for ts in time_steps:
        yield (ts, val)
        if cnt == interval:
            if val == 1: val = -1
            else: val = 1
            interval = points_per_square + np.random.random_integers(0, square_jiggle_offsets)
            cnt = 0
        cnt += 1


def spikes(x, offset, points_per_jig, spike_jiggle_offsets):
    times_pi = x / points_per_jig
    time_steps = np.linspace(0, times_pi * np.pi, x)
    time_steps = time_steps + offset
    val = 1
    cnt = 0
    interval = points_per_jig + np.random.random_integers(0, spike_jiggle_offsets)
    for ts in time_steps:
        yield (ts, ((cnt/points_per_jig)*2-1)*val)
        cnt += 1
        if cnt == interval:
            cnt = 0
            val = val*-1
            interval = points_per_jig + np.random.random_integers(0, spike_jiggle_offsets)


def flat_plus_peaks(x, offset, scale_for_pi):
    times_pi = x / scale_for_pi
    num_diracs = int(times_pi) # idea: we want as many peaks as we have pi segments
    time_steps = np.linspace(0, times_pi * np.pi, x)
    time_steps = time_steps + offset
    dirac_ts = np.random.choice(time_steps, num_diracs, replace=False)
    val = 1
    for ts in time_steps:
        if ts in dirac_ts:
            val = val * -1
            yield (ts, val)
        else:
            yield (ts, 0)


def recompute_time(aug, sensor_resolution):
    d = np.copy(aug.data)
    n = len(d)
    times_pi = n / sensor_resolution
    time_steps = np.linspace(0, times_pi * np.pi, n)
    d[:,0] = time_steps
    return d


class SignalTypes:

    TYPES = ['sinus', 'square', 'spikes', 'flat_plus_peaks']

    def __init__(self, sensor_resolution):
        self.sensor_resolution = sensor_resolution

    def get_callback_dict(self, selected_types, spike_jiggle_offsets, square_jiggle_offsets):
        sin_callback = functools.partial(sinus, points_per_pi=self.sensor_resolution)
        square_callback = functools.partial(square, points_per_square=self.sensor_resolution, square_jiggle_offsets=square_jiggle_offsets)
        spikes_callback = functools.partial(spikes, points_per_jig=self.sensor_resolution, spike_jiggle_offsets=spike_jiggle_offsets)
        fpp_callback = functools.partial(flat_plus_peaks, scale_for_pi=self.sensor_resolution)
        ts_generators = {
            'sinus': sin_callback,
            'square': square_callback,
            'spikes': spikes_callback,
            'flat_plus_peaks': fpp_callback
        }
        selected_generators = {k: v for k, v in ts_generators.items() if k in selected_types}
        return selected_generators


class TimeSeriesCreator:  # for now a Creator, maybe later a real python generator, if I am less lazy

    def __init__(self, ts_length, ts_segments_base_length, ts_types, ts_type_generators):
        self.ts_length = ts_length
        self.ts_segments_base_length = ts_segments_base_length
        self.ts_types = ts_types
        self.ts_type_generators = ts_type_generators
        self.segements = []
        self._create_ts()

    def _create_ts(self):
        segments = None
        segment_lengths = random_segments(self.ts_length, self.ts_segments_base_length)
        last_type = None
        segment_offset = 0
        for s_len in segment_lengths:
            t = rem_last_type(self.ts_types, last_type)
            segment, last_type = create_time_series_segment(s_len, segment_offset, t, self.ts_type_generators)
            s_np = np.array(segment)
            segment_offset = s_np[-1,0]
            if segments is None:
                segments = s_np
            else:
                segments = np.append(segments, s_np, axis=0)
        self.segements = segments

    @property
    def ts(self):
        return self.segements


class SinusGenerator:

    def __init__(self, times_pi, points_per_pi, mu, sigma):
        self.times_pi = times_pi
        self.points_per_pi = points_per_pi
        self.mu = mu
        self.sigma = sigma

    def _comp_shown_seq_length(self):
        shown_seq_length = self.times_pi * self.points_per_pi
        return shown_seq_length

    def generate_basic(self):
        shown_seq_length = self._comp_shown_seq_length()
        time_steps = np.linspace(0, self.times_pi * np.pi, shown_seq_length)
        data = np.sin(time_steps)
        res = np.column_stack((time_steps, data))
        return res

    def generate_noise(self):
        shown_seq_length = self._comp_shown_seq_length()
        base = self.generate_basic()
        time_steps = np.copy(base[:,0])
        noise = np.random.normal(self.mu, self.sigma, shown_seq_length)
        noisy = base[:,1] + noise
        res = np.column_stack((time_steps, noisy))
        return (res, base, noise)

    def linespace(self, number_samples):
        time_steps = np.linspace(0, self.times_pi * np.pi, number_samples)
        return time_steps


class Sensor:

    def __init__(self, sampling_rate, noise_mu, noise_sigma):
        self.sampling_rate = sampling_rate
        self.noise_mu = noise_mu
        self.noise_sigma = noise_sigma

    def sense_signal(self, ts_length, segment_base_length=None, ts_types=None, spike_jiggle_offsets=3, square_jiggle_offsets=10):
        real_ts_length = int((ts_length/100)*self.sampling_rate)
        if segment_base_length is None:
            segment_base_length = self.sampling_rate*2
        if ts_types is None:
            ts_types = SignalTypes.TYPES
        ts_generators = SignalTypes(self.sampling_rate).get_callback_dict(ts_types, spike_jiggle_offsets, square_jiggle_offsets)
        tsc = TimeSeriesCreator(real_ts_length, segment_base_length, ts_types, ts_generators)
        noisy = AugmentTSSignal(tsc.ts).add_noise(self.noise_mu, self.noise_sigma).data

        return tsc.ts, noisy


class AugmentTSSignal:

    def __init__(self, data, other=None, expand_from=None, expand_to=None, remove_every=None,
                 noise_mu=None, noise_sigma=None, noise_op=None, block_offset=None, block_length=None):
        self.data = data
        self._set_field('expand_from', expand_from, other)
        self._set_field('expand_to', expand_to, other)
        self._set_field('remove_every', remove_every, other)
        self._set_field('noise_mu', noise_mu, other)
        self._set_field('noise_sigma', noise_sigma, other)
        self._set_field('noise_op', noise_op, other)
        self._set_field('block_offset', block_offset, other)
        self._set_field('block_length', block_length, other)

    def _set_field(self, field_name, new_field, other):
        if new_field is not None:
            setattr(self, field_name, new_field)
            return
        other_attr = getattr(other, field_name, None)
        setattr(self, field_name, other_attr)

    def expand(self, expand_from, expand_to, sensor_resolution=None):
        d = basic_representation(self.data, expand_from, expand_to)
        res = AugmentTSSignal(d, other=self, expand_from=expand_from, expand_to=expand_to)
        if sensor_resolution is not None:
            cb = functools.partial(recompute_time, sensor_resolution=sensor_resolution)
            res = res.apply_callback(cb)
        return res

    def downsample(self, remove_every, is_all_data=False): # some also say resize do that
        d = downsample(self.data, remove_every, is_all_data)
        return AugmentTSSignal(d, other=self, remove_every=remove_every)

    def op_noise(self, mu, sigma, op):
        idx = get_data_idx(self.data)
        noise_len = len(idx)
        noise = np.random.normal(mu, sigma, noise_len)
        d = np.copy(self.data)
        d[idx,1] = op(d[idx,1], noise)
        return AugmentTSSignal(d, other=self, noise_mu=mu, noise_sigma=sigma, noise_op=op)

    def add_noise(self, mu, sigma):
        return self.op_noise(mu, sigma, operator.add)

    def sub_moise(self, mu, sigma):
        return self.op_noise(mu, sigma, operator.sub)

    def random_block(self, min_length, max_length):
        block_length = int(np.random.uniform(min_length, max_length, 1))
        block_offset = int(np.random.uniform(0, len(self.data)-block_length))
        return block_offset, block_length

    def rem_block(self, offset, length):
        idx = get_data_idx(self.data)
        rem_idx = idx[offset:offset+length]
        d = np.copy(self.data)
        if d.shape[1] == 2:
            d = add_mark_missing_column(d)
        d[rem_idx, -2] = 0.
        d[rem_idx, -1] = 1
        return AugmentTSSignal(d, other=self, block_offset=offset, block_length=length)

    def apply_callback(self, callback):
        d = callback(self)
        return AugmentTSSignal(d)

