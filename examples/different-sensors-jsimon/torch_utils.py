import torch
from torch.utils.data import Dataset
import numpy as np
import utils as ut


class BadSensorsDataset(Dataset):
    """Bad Sensors dataset."""

    def __init__(self, sensors, sample_steps, num_samples_per_sensor, jiggle_offsets=None, transform=None, return_two_transforms=True):
        """
        Total data generated (ignoring transforms) is len(sensors) * sample_steps * num_samples_per_sensor

        Args:
            sensors (list): list of sensors to create time series
            sample_steps (int): number of time steps for one sample, normalised for 100Hz
            num_samples_per_sensor (int): number of samples to generate
            jiggle_offsets (int or None): if given jiggle the offset of a sample uniform
                randomly in 0 to jiggle_offsets
            transform (callable, optional): Optional transform to be applied
                on a sample.
            return_two_transforms (bool): Flag to indicate if two different samples with two
                different transform should be returned
        """
        self.sensors = sensors
        self.sample_steps = sample_steps
        self.num_samples_per_sensor = num_samples_per_sensor
        self.jiggle_offsets = jiggle_offsets

        self.signal = []
        ts_length_in100Hz = self.ts_length
        for sensor in sensors:
            signal, signal_w_noise = sensor.sense_signal(ts_length_in100Hz)
            signal_100hz = ut.AugmentTSSignal(signal_w_noise).expand(sensor.sampling_rate, 100, sensor.sampling_rate).data
            assert len(signal_100hz) == ts_length_in100Hz
            self.signal.append(signal_100hz)

        self.return_two_transforms = return_two_transforms
        self.transform = transform

    def __len__(self):
        return len(self.sensors) * self.num_samples_per_sensor

    @property
    def ts_length(self):
        return self.sample_steps * self.num_samples_per_sensor

    def get_single_item(self, idx):
        sensor_idx = idx // self.num_samples_per_sensor
        sample_idx = idx % self.num_samples_per_sensor
        time_series = self.signal[sensor_idx]
        start = self.sample_steps * sample_idx
        if self.jiggle_offsets is not None:
            if idx - self.sample_steps > start:
                start += np.random.randint(0, self.jiggle_offsets)
        end = start + self.sample_steps
        #print(end-start, start, end)
        sample = time_series[start:end]
        return sample

    def apply_transforms(self, sample):
        if self.transform:
            if self.return_two_transforms:
                s1 = self.transform(sample)
                s2 = self.transform(sample)
                sample = (s1, s2)
            else:
                sample = self.transform(sample)
        return sample

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if isinstance(idx, slice):
            idx = [self[ii] for ii in range(*idx.indices(len(self)))]

        if isinstance(idx, list):
            samples = [self.get_single_item(i) for i in idx]
            samples = [self.apply_transforms(s) for s in samples]
            return samples

        # else we assume it is a single index so:
        sample = self.get_single_item(idx)
        sample = self.apply_transforms(sample)

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        return torch.from_numpy(sample)


class RandomDownsample(object):
    """Randomly downsampling by removing samples

    Args:
        min_factor (int): the minimum number to remove every factor entry of data
    """

    def __init__(self, min_factor=10):
        self.min_factor = min_factor

    def __call__(self, sample):
        factor = np.random.randint(0, self.min_factor)
        if factor == 0:
            return np.copy(sample)
        sample = ut.AugmentTSSignal(sample).downsample(factor).data
        return sample


class AddNoise(object):
    """Randomly add noise from a gaussian distribuiton to the signal

    Args:
        mu (float or tuple): the mean of the gaussian for the noise
        sigma (float or tuple): the variance of the gaussian of the noise
    """

    def __init__(self, mu=0.0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, sample):
        mu = self.mu
        sigma = self.sigma
        if isinstance(mu, tuple):
            mu = np.random.uniform(mu[0], mu[1])
        if isinstance(sigma, tuple):
            sigma = np.random.uniform(sigma[0], sigma[1])
        sample = ut.AugmentTSSignal(sample).add_noise(mu, sigma).data
        return sample


class AblateBlock(object):
    """Randomly ablate a block of the time series

    Args:
        min_length (int): minimum length of ablation
        max_length (int): maximum length of ablation
    """

    def __init__(self, min_length, max_length):
        self.min_length = min_length
        self.max_length = max_length

    def __call__(self, sample):
        augmentor = ut.AugmentTSSignal(sample)
        block_offset, block_length = augmentor.random_block(self.min_length, self.max_length)
        sample = augmentor.rem_block(block_offset, block_length).data
        return sample
