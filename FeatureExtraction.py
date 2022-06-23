import librosa
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import dct 
import scipy.io.wavfile
import csv



def mfcc(audio_file, n_mfccs):
    sample_rate, signal = scipy.io.wavfile.read(audio_file)
    
    pre_emphasis = 0.97 # tien xu ly
    emphasized_signal = emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
    
    
    # chia frame
    frame_stride = 0.01 # độ chèn của các frame
    frame_size = 0.025 # kích cỡ frame

    frame_length = frame_size * sample_rate #chuyển đổi từ second sang samples
    frame_step = frame_stride * sample_rate #chuyển đổi từ second sang samples
    signal_length = len(emphasized_signal) ##
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))#đảm bảo luôn có ít nhất 1 khung
    
    pad_signal_length = num_frames * frame_step + frame_length ##
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(emphasized_signal, z) ## Pad Signal để đảm bảo rằng tất cả các khung đều có số lượng mẫu bằng nhau mà không cắt bớt bất kỳ mẫu nào từ tín hiệu ban đầu


    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    frames *= np.hamming(frame_length) # frames *= 0.54 - 0.46 * numpy.cos((2 * numpy.pi * n) / (frame_length - 1))  # Explicit Implementation **
    
    # FFT convert
    NFFT = 512 
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))
    
    
    # mel filter
    nfilt = 40
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2) # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = np.floor((NFFT + 1) * hz_points / sample_rate)

    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
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
    
    # dct
    num_ceps = n_mfccs
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)] # Keep 2-13
    mfcc = np.mean(mfcc, axis=0)
    
    return mfcc



audio_folder = ["Banjo", "Cello", "DoubleBass", "ElectricBass", "Guitar", "Harp", "Mandolin", "Ukelele", "Viola", "Violin"]


for i in range(0, len(audio_folder)):
    instrument = audio_folder[i]
    audio_type = "\\" + instrument
    audio_link = "F:\\Uni\\4\\Spring\\CSDL-DPT\\Audio"
    
    linkAudioArr = [] # get link file audio
    for i in range(1, 16):
        audio_full = audio_link + audio_type + audio_type + str(i) + ".wav"
        linkAudioArr.append(audio_full)
        print(audio_full)
    
    listFeature = [] #
    for item in linkAudioArr:
        sample_rate, signal = scipy.io.wavfile.read(item)
        mfccs = mfcc(item, 13)
        arr = mfccs
        arr = np.append(arr, instrument)
        listFeature.append(arr)
    
    fileCSV = open("F:\\Uni\\4\\Spring\\CSDL-DPT\\code\\BTL\\FeatureExtraction1.csv", "a", newline="")
    writer = csv.writer(fileCSV)
    writer.writerows(listFeature)
    fileCSV.close()