import torch
import torchaudio
import librosa


class ACPower(object):

    def __init__(self, net_frequency=50, measurement_frequency=6400):
        """ Calculates Real, Apparent, Reactive and Distortion Power.

        Args:
            net_frequency (int): Frequency of the net 50Hz or 60Hz
            measurement_frequency (int): Frequency of the measurements
        """
        self.cycle_length = int(measurement_frequency / net_frequency)
        self.feature_dim = 4

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

    def __init__(self, net_frequency=50, measurement_frequency=6400):
        """ Calculates Device Current Signature (DCS).

        Args:
            net_frequency (int): Frequency of the net 50Hz or 60Hz
            measurement_frequency (int): Frequency of the measurements
        """
        self.cycle_length = int(measurement_frequency / net_frequency)
        self.feature_dim = 1

    def __call__(self, sample):

        current_in, voltage_in, features, target = sample

        cycle_n = int(current_in.size(0) / self.cycle_length)

        current = current_in.reshape(cycle_n, self.cycle_length)

        current_peaks = torch.amax(current, dim=1)

        assert (cycle_n % 2 == 0)

        current_peaks_a, current_peaks_b = torch.split(current_peaks, int(cycle_n / 2))

        # Relative magnitude of the current peaks (ON/OFF events)
        mean_steady_peak_a = torch.mean(current_peaks_a)
        mean_steady_peak_b = torch.mean(current_peaks_b)

        # device_current_signature = current_peaks - mean_steady_peak
        device_current_signature = torch.zeros(cycle_n)
        device_current_signature[:int(cycle_n / 2)] = current_peaks_a - mean_steady_peak_b
        device_current_signature[int(cycle_n / 2):] = current_peaks_b - mean_steady_peak_a

        if features is None:
            features = device_current_signature.unsqueeze(0)
        else:
            features = torch.vstack([features, device_current_signature])

        return current_in, voltage_in, features, target


class COT(object):

    def __init__(self, net_frequency=50, measurement_frequency=6400):
        """ Calculates Current Over Time (COT).

        Args:
            net_frequency (int): Frequency of the net 50Hz or 60Hz
            measurement_frequency (int): Frequency of the measurements
        """
        self.cycle_length = int(measurement_frequency / net_frequency)
        self.feature_dim = 1

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

    def __init__(self, net_frequency=50, measurement_frequency=6400):
        """ Calculates Admittance Over Time (AOT).

        Args:
            net_frequency (int): Frequency of the net 50Hz or 60Hz
            measurement_frequency (int): Frequency of the measurements
        """
        self.cycle_length = int(measurement_frequency / net_frequency)
        self.feature_dim = 1

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

    def __init__(self, net_frequency=50, measurement_frequency=6400):
        """ Calculates the Mel Spectrogram of an event.

        Args:
            net_frequency (int): Frequency of the net 50Hz or 60Hz
            measurement_frequency (int): Frequency of the measurements
        """
        self.n_fft = int(measurement_frequency / net_frequency) * 2 - 1
        self.hop_length = int((self.n_fft / 2) + 1)
        self.torch_spec = torchaudio.transforms.Spectrogram(n_fft=self.n_fft, hop_length=self.hop_length)
        self.feature_dim = self.torch_spec.hop_length

    def __call__(self, sample):
        current_in, voltage_in, features, target = sample

        # Try if PyTorch is installed with working fft, else use librosa library
        try:
            spec = self.torch_spec(current_in)
        except RuntimeError:
            spec, _ = librosa.core.spectrum._spectrogram(current_in.numpy(),
                                                         n_fft=self.n_fft,
                                                         hop_length=self.hop_length,
                                                         center=True,
                                                         pad_mode="reflect",
                                                         power=2.0)
            spec = torch.as_tensor(spec)

        if features is None:
            features = spec
        else:
            features = torch.vstack([features, spec])

        return current_in, voltage_in, features, target


class MelSpectrogram(object):

    def __init__(self, net_frequency=50, measurement_frequency=6400):
        """ Calculates the Mel Spectrogram of an event.

        Args:
            net_frequency (int): Frequency of the net 50Hz or 60Hz
            measurement_frequency (int): Frequency of the measurements
        """
        self.measurement_frequency = measurement_frequency
        self.n_fft = int(measurement_frequency / net_frequency) * 2 - 1
        self.hop_length = int((self.n_fft / 2) + 1)
        self.torch_mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate=measurement_frequency, n_fft=self.n_fft,
                                                                   hop_length=self.hop_length, n_mels=64)
        self.feature_dim = self.torch_mel_spec.n_mels

    def __call__(self, sample):
        current_in, voltage_in, features, target = sample
        current_in = current_in.float()

        # Try if PyTorch is installed with working fft, else use librosa library
        try:
            mel_spec = self.torch_mel_spec(current_in)
        except RuntimeError:
            mel_spec = librosa.feature.melspectrogram(current_in.numpy(),
                                                      sr=self.measurement_frequency,
                                                      n_fft=self.n_fft,
                                                      hop_length=self.hop_length,
                                                      center=True,
                                                      pad_mode="reflect",
                                                      power=2.0,
                                                      n_mels=64,
                                                      norm=None,
                                                      htk=True)
            mel_spec = torch.as_tensor(mel_spec)

        if features is None:
            features = mel_spec
        else:
            features = torch.vstack([features, mel_spec])

        return current_in, voltage_in, features, target


class MFCC(object):

    def __init__(self, net_frequency=50, measurement_frequency=6400):
        """ Calculates the  Mel-frequency cepstrum coefficients of an event.

        Args:
            net_frequency (int): Frequency of the net 50Hz or 60Hz
            measurement_frequency (int): Frequency of the measurements
        """
        self.measurement_frequency = measurement_frequency
        self.n_fft = int(measurement_frequency / net_frequency) * 2 - 1
        self.hop_length = int((self.n_fft / 2) + 1)
        self.torch_mfcc = torchaudio.transforms.MFCC(sample_rate=measurement_frequency, n_mfcc=64,
                                                     melkwargs={"n_fft": self.n_fft,
                                                                "hop_length": self.hop_length,
                                                                'n_mels': 64})
        self.feature_dim = self.torch_mfcc.n_mfcc

    def __call__(self, sample):
        current_in, voltage_in, features, target = sample
        current_in = current_in.float()

        # Try if PyTorch is installed with working fft, else use librosa library
        try:
            mfcc = self.torch_mfcc(current_in)
        except RuntimeError:
            mfcc = librosa.feature.mfcc(y=current_in.numpy(),
                                        sr=self.measurement_frequency,
                                        n_mfcc=64,
                                        hop_length=self.hop_length,
                                        n_fft=self.n_fft,
                                        n_mels=64,
                                        htk=True)

            mfcc = torch.as_tensor(mfcc)

        if features is None:
            features = mfcc
        else:
            features = torch.vstack([features, mfcc])

        return current_in, voltage_in, features, target


class RandomAugment(object):

    def __init__(self, net_frequency=50, measurement_frequency=6400, p=0.8, augment_i=-1):
        """ Randomly applies an augmentation on the window. PhaseShift (Left, Right) or HalfPhaseFlip (Left, Right).

        Args:
            net_frequency (int): Frequency of the net 50Hz or 60Hz
            measurement_frequency (int): Frequency of the measurements
            p (float): Probability of an augmentation being applied on the window
            augment_i (int): Used for manual augmentation selection
        """
        self.cycle_length = int(measurement_frequency / net_frequency)
        self.p = p
        self.feature_dim = 0
        self.augment_i = augment_i

    def __call__(self, sample):

        current, voltage, features, target = sample

        if self.p > 0 and self.augment_i == -1:
            i = int(4 / self.p)
            self.augment_i = (torch.randint(i, (1,))).item()

        idx_0 = self.cycle_length
        idx_1 = self.cycle_length * 2
        idx_2 = int(1.5 * self.cycle_length)
        idx_3 = int(self.cycle_length / 2)

        # Phase Shift Left
        if self.augment_i == 0:
            current = current[idx_1:]
            voltage = voltage[idx_1:]

        # Phase Shift Right
        elif self.augment_i == 1:
            current = current[:current.size(0) - idx_1]
            voltage = voltage[:voltage.size(0) - idx_1]

        # Half Phase Flip Left
        elif self.augment_i == 2:
            current = current[idx_2:current.size(0) - idx_3] * -1
            voltage = voltage[idx_2:voltage.size(0) - idx_3] * -1

        # Half Phase Flip Right
        elif self.augment_i == 3:
            current = current[idx_3:current.size(0) - idx_2] * -1
            voltage = voltage[idx_3:voltage.size(0) - idx_2] * -1

        # Return unaugmented window
        else:
            current = current[idx_0:current.size(0) - idx_0]
            voltage = voltage[idx_0:voltage.size(0) - idx_0]

        return current, voltage, features, target


if __name__ == '__main__':

    current = torch.rand(24576)
    voltage = torch.rand(24576)
    features = None

    from torchvision import transforms

    measurement_frequency = 6400
    transform = transforms.Compose([
        RandomAugment(measurement_frequency=measurement_frequency),
        ACPower(measurement_frequency=measurement_frequency),
        DCS(measurement_frequency=measurement_frequency),
        COT(measurement_frequency=measurement_frequency),
        AOT(measurement_frequency=measurement_frequency)
    ])
    for i in range(0, 10):
        x = transform((current, voltage, features, [0]))
        print(x[2].shape)

    transform = transforms.Compose([
        RandomAugment(measurement_frequency=measurement_frequency),
        Spectrogram(measurement_frequency=measurement_frequency),
        MelSpectrogram(measurement_frequency=measurement_frequency),
        MFCC(measurement_frequency=measurement_frequency)
    ])
    for i in range(0, 10):
        x = transform((current, voltage, features, [0]))
        print(x[2].shape)
