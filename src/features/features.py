import numpy as np
import torch
import torchaudio


class ACPower(object):

    def __init__(self, net_frequency=50, measurement_frequency=50000):
        """ Calculates Real, Apparent, Reactive and Distortion Power.

        Args:
            net_frequency (int): Frequency of the net 50Hz or 60Hz
            measurement_frequency (int): Frequency of the measurements
        """
        self.cycle_length = int(measurement_frequency / net_frequency)

    def __call__(self, sample):

        current_in, voltage_in, features, target = sample

        cycle_n = int(current_in.size(0) / self.cycle_length)

        current = current_in.reshape(cycle_n, self.cycle_length)
        voltage = voltage_in.reshape(cycle_n, self.cycle_length)

        # P = Sum(i[n] * v[n]) / N)
        active_power = torch.sum(current * voltage / self.cycle_length, dim=1)

        current_rms = torch.sqrt(torch.mean(current ** 2, dim=1))
        voltage_rms = torch.sqrt(torch.mean(voltage ** 2, dim=1))

        # S = Irms * Vrms
        apparent_power = current_rms * voltage_rms

        # Q = sqrt(S^2 - P^2)
        reactive_power = torch.sqrt(apparent_power ** 2 - active_power ** 2)

        # D = sqrt(S^2 - Q^2 - P^2)
        distortion_power = torch.sqrt(torch.abs(apparent_power ** 2 - reactive_power ** 2 - active_power ** 2))

        if features is None:
            features = torch.vstack([active_power, apparent_power, reactive_power, distortion_power])
        else:
            features = torch.vstack([features, active_power, apparent_power, reactive_power, distortion_power])

        return current_in, voltage_in, features, target


class DCS(object):

    def __init__(self, net_frequency=50, measurement_frequency=50000):
        """ Calculates Device Current Signature (DCS).

        Args:
            net_frequency (int): Frequency of the net 50Hz or 60Hz
            measurement_frequency (int): Frequency of the measurements
        """
        self.cycle_length = int(measurement_frequency / net_frequency)

    def __call__(self, sample):

        current_in, voltage_in, features, target = sample

        cycle_n = int(current_in.size(0) / self.cycle_length)

        current = current_in.reshape(cycle_n, self.cycle_length)

        current_peaks = torch.amax(current, dim=1)

        assert (cycle_n == 50 or cycle_n == 90 or cycle_n == 25)
        # Split for 0.5 and 1 second window
        if cycle_n == 50:
            current_peaks_A, current_peaks_B = torch.split(current_peaks, 25)

        elif cycle_n == 25:
            current_peaks_A, current_peaks_B = torch.split(current_peaks, 13)

        # Split for 1.5 second windows
        elif cycle_n == 90:
            tmp = np.split(current_peaks, 3)
            current_peaks_A = tmp[0]
            current_peaks_B = torch.append(tmp[1], tmp[2])

        # Relative magnitude of the current peaks
        mean_steady_peak_A = torch.mean(current_peaks_A)

        # device_current_signature = current_peaks_B - mean_steady_peak_A
        device_current_signature = current_peaks - mean_steady_peak_A

        if features is None:
            features = device_current_signature
        else:
            features = torch.vstack([features, device_current_signature])

        return current_in, voltage_in, features, target


class COT(object):

    def __init__(self, net_frequency=50, measurement_frequency=50000):
        """ Calculates Current Over Time (COT).

        Args:
            net_frequency (int): Frequency of the net 50Hz or 60Hz
            measurement_frequency (int): Frequency of the measurements
        """
        self.cycle_length = int(measurement_frequency / net_frequency)

    def __call__(self, sample):
        current_in, voltage_in, features, target = sample

        cycle_n = int(current_in.size(0) / self.cycle_length)

        current = current_in.reshape(cycle_n, self.cycle_length)

        current_rms = torch.sqrt(torch.mean(current ** 2, dim=1))

        if features is None:
            features = current_rms.unsqueeze(0)
        else:
            features = torch.vstack([features, current_rms])

        return current_in, voltage_in, features, target


class AOT(object):

    def __init__(self, net_frequency=50, measurement_frequency=50000):
        """ Calculates Admittance Over Time (AOT).

        Args:
            net_frequency (int): Frequency of the net 50Hz or 60Hz
            measurement_frequency (int): Frequency of the measurements
        """
        self.cycle_length = int(measurement_frequency / net_frequency)

    def __call__(self, sample):
        current_in, voltage_in, features, target = sample

        cycle_n = int(current_in.size(0) / self.cycle_length)

        current = current_in.reshape(cycle_n, self.cycle_length)
        voltage = voltage_in.reshape(cycle_n, self.cycle_length)

        current_rms = torch.sqrt(torch.mean(current ** 2, dim=1))
        voltage_rms = torch.sqrt(torch.mean(voltage ** 2, dim=1))

        aot = current_rms / voltage_rms

        if features is None:
            features = aot.unsqueeze(0)
        else:
            features = torch.vstack([features, aot])

        return current_in, voltage_in, features, target


class Spectrogram(object):

    def __init__(self, n_fft=2000):
        """ Calculates the Mel Spectrogram of an event.

        Args:
            n_fft: Size of FFT, creates n_fft // 2 + 1 bins.
        """
        self.torch_spec = torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=1001)

    def __call__(self, sample):
        current_in, voltage_in, features, target = sample

        spec = self.torch_spec(current_in)

        if features is None:
            features = spec
        else:
            features = torch.vstack([features, spec])

        return current_in, voltage_in, features, target


class MelSpectrogram(object):

    def __init__(self, n_fft=2000, measurement_frequency=50000):
        """ Calculates the Mel Spectrogram of an event.

        Args:
            n_fft: Size of FFT, creates n_fft // 2 + 1 bins.
            measurement_frequency (int): Frequency of the measurements
        """
        self.torch_mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate=measurement_frequency, n_fft=n_fft,
                                                                   hop_length=1001)

    def __call__(self, sample):
        current_in, voltage_in, features, target = sample
        current_in = current_in.float()

        mel_spec = self.torch_mel_spec(current_in)

        if features is None:
            features = mel_spec
        else:
            features = torch.vstack([features, mel_spec])

        return current_in, voltage_in, features, target


class MFCC(object):

    def __init__(self, n_fft=2000, measurement_frequency=50000):
        """ Calculates the  Mel-frequency cepstrum coefficients of an event.

        Args:
            n_fft: Size of FFT, creates n_fft // 2 + 1 bins.
            measurement_frequency (int): Frequency of the measurements
        """
        self.torch_mfcc = torchaudio.transforms.MFCC(sample_rate=measurement_frequency, n_mfcc=64,
                                                     melkwargs={"n_fft": n_fft, "hop_length": 1001})

    def __call__(self, sample):
        current_in, voltage_in, features, target = sample
        current_in = current_in.float()

        mfcc = self.torch_mfcc(current_in)

        if features is None:
            features = mfcc
        else:
            features = torch.vstack([features, mfcc])

        return current_in, voltage_in, features, target


class RandomAugment(object):

    def __init__(self, net_frequency=50, measurement_frequency=50000, p_augment=0.8):
        """ Randomly applies an augmentation on the window. PhaseShift (Left, Right) or HalfPhaseFlip (Left, Right).

        Args:
            net_frequency (int): Frequency of the net 50Hz or 60Hz
            measurement_frequency (int): Frequency of the measurements
            p_augment (float): Probability of an augmentation being applied on the window
        """
        self.cycle_length = int(measurement_frequency / net_frequency)
        self.p_augment = p_augment

    def __call__(self, sample):

        current, voltage, features, target = sample

        i = int(4 / self.p_augment)
        augment_i = (torch.randint(i, (1,))).item()

        idx_0 = self.cycle_length
        idx_1 = self.cycle_length * 2
        idx_2 = int(1.5 * self.cycle_length)
        idx_3 = int(self.cycle_length / 2)

        # Phase Shift Left
        if augment_i == 0:
            current = current[idx_1:]
            voltage = voltage[idx_1:]

        # Phase Shift Right
        elif augment_i == 1:
            current = current[:current.size(0) - idx_1]
            voltage = voltage[:voltage.size(0) - idx_1]

        # Half Phase Flip Left
        elif augment_i == 2:
            current = current[idx_2:current.size(0) - idx_3] * -1
            voltage = voltage[idx_2:voltage.size(0) - idx_3] * -1

        # Half Phase Flip Right
        elif augment_i == 3:
            current = current[idx_3:current.size(0) - idx_2] * -1
            voltage = voltage[idx_3:voltage.size(0) - idx_2] * -1

        # Return unaugmented window
        else:
            current = current[idx_0:current.size(0) - idx_0]
            voltage = voltage[idx_0:voltage.size(0) - idx_0]

        return current, voltage, features, target


if __name__ == '__main__':

    current = torch.rand(52000)
    voltage = torch.rand(52000)
    features = None

    from torchvision import transforms


    transform = transforms.Compose([
        RandomAugment(),
        ACPower(),
        DCS(),
        COT(),
        AOT()
    ])
    for i in range(0, 10):
        x = transform((current, voltage, features, [0]))
        print(x[2].shape)


    transform = transforms.Compose([
        RandomAugment(),
        Spectrogram(),
        MelSpectrogram(),
        MFCC()
    ])
    for i in range(0, 10):
        x = transform((current, voltage, features, [0]))
        print(x[2].shape)
