import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct

class Features:
    PRE_EMPHASIS = 0.97
    FRAME_SIZE = 0.025
    FRAME_STRIDE = 0.01
    NFFT = 512
    NO_FILTERS = 40
    
    def __init__(self, input, rate):
        self.input = input
        self.rate = rate
    
    def pad_array_if_necessary(self, constant_value_left = 0, constant_value_right = 0):
        if(len(self.input) < self.rate):
            pad_lenght = int((self.rate - len(self.input))/2)
            self.input = np.pad(array = self.input, pad_width = (pad_lenght, pad_lenght), 
                           mode = 'constant', constant_values = (constant_value_left, constant_value_right))
            if(len(self.input) + 1 == self.rate):
                self.input = np.append(arr = self.input, values = 0)
        else:
            self.input = self.input[0:self.rate]

    def emphasising_the_singnal(self):
        self.input = np.append(self.input[0], self.input[1:] - Features.PRE_EMPHASIS * self.input[:-1])
        
    def framing_the_signal(self):
        frame_length, frame_step = Features.FRAME_SIZE * self.rate, Features.FRAME_STRIDE * self.rate  # Convert from seconds to samples
        signal_length = len(self.input)
        frame_length = int(round(frame_length))
        frame_step = int(round(frame_step))
        num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame
        
        pad_signal_length = num_frames * frame_step + frame_length
        zeros = np.zeros((pad_signal_length - signal_length))
        pad_signal = np.append(self.input, zeros) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal
        
        indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
        frames = pad_signal[indices.astype(np.int32, copy=False)]
        return frames
    
    def fourier_transform(self, frames):
        mag_frames = np.absolute(np.fft.rfft(frames, Features.NFFT))  # Magnitude of the FFT
        pow_frames = ((1.0 / Features.NFFT) * ((mag_frames) ** 2))
        return mag_frames, pow_frames
    
    def create_filtre_banks(self, mag_frames, pow_frames):
        low_freq_mel = 0
        high_freq_mel = (2595 * np.log10(1 + (self.rate / 2) / 700))  # Convert Hz to Mel
        mel_points = np.linspace(low_freq_mel, high_freq_mel, Features.NO_FILTERS + 2)  # Equally spaced in Mel scale
        hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
        bin = np.floor((Features.NFFT + 1) * hz_points / self.rate)
        
        fbank = np.zeros((Features.NO_FILTERS, int(np.floor(Features.NFFT / 2 + 1))))
        for m in range(1, Features.NO_FILTERS + 1):
            f_m_minus = int(bin[m - 1])   # left
            f_m = int(bin[m])             # center
            f_m_plus = int(bin[m + 1])    # right
        
            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
        filter_banks = np.dot(pow_frames, fbank.T)
        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
        filter_banks = 20 * np.log10(filter_banks)  # dB
        return filter_banks

    def plot_results(self, result, xlabel = 'Time', ylabel = 'Frequency'):
        plt.imshow(result.T)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

    def wav_to_features(self, feature_type = 'filtrebanks', plot = 0):
        result = np.zeros(0)
        if(feature_type == 'filtrebanks'):
            self.pad_array_if_necessary()
            self.emphasising_the_singnal()
            frames = self.framing_the_signal()
            mag_frames, pow_frames = self.fourier_transform(frames)
            result = self.create_filtre_banks(mag_frames, pow_frames)
        if(plot == 1):
            self.plot_results(result = result)
        return result