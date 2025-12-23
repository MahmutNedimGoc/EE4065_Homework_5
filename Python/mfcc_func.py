import numpy as np
import scipy.signal as sig
from scipy.fftpack import dct
import scipy.io.wavfile as wav

def create_mfcc_features(recordings_list, FFTSize, sample_rate, numOfMelFilters, numOfDctOutputs):
    feature_list = []
    label_list = []
    
    for recording in recordings_list:
        file_path = recording[0] + "/" + recording[1]
        
        try:
            # Ses dosyasını oku
            rate, data = wav.read(file_path)
            
            # Stereo ise mono yap
            if len(data.shape) > 1:
                data = data[:, 0]
            data = data.astype(float)
        except Exception as e:
            # Okuma hatası olursa atla
            print(f"Dosya okunamadı: {file_path} - Hata: {e}")
            continue

        # --- İŞLEMLER ---
        
        # 1. Ön Vurgu (Pre-emphasis)
        pre_emphasis = 0.97
        emphasized_signal = np.append(data[0], data[1:] - pre_emphasis * data[:-1])
        
        # 2. Çerçeveleme (Framing) AYARLARI
        frame_stride = 0.01
        frame_length_sec = 0.025 
        
        frame_length = int(round(frame_length_sec * sample_rate))
        frame_step = int(round(frame_stride * sample_rate))
        
        signal_length = len(emphasized_signal)
        num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))
        
        pad_signal_length = num_frames * frame_step + frame_length
        z = np.zeros((pad_signal_length - signal_length))
        pad_signal = np.append(emphasized_signal, z)
        
        # --- HATA BURADAYDI, DÜZELTİLDİ ---
        # Matris boyutlarını eşitlemek için güvenli yöntem:
        indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + \
                  np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
        
        # Olası taşmaları engellemek için float32'ye çevirip integer yapıyoruz
        indices = np.array(indices, dtype=np.int32)
        
        frames = pad_signal[indices]
        
        # 3. Pencereleme
        frames *= np.hamming(frame_length)
        
        # 4. Fourier ve Güç Spektrumu
        mag_frames = np.absolute(np.fft.rfft(frames, FFTSize))
        pow_frames = ((1.0 / FFTSize) * ((mag_frames) ** 2))
        
        # 5. Mel Filtre Bankası
        low_freq_mel = 0
        high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))
        mel_points = np.linspace(low_freq_mel, high_freq_mel, numOfMelFilters + 2)
        hz_points = (700 * (10**(mel_points / 2595) - 1))
        bin = np.floor((FFTSize + 1) * hz_points / sample_rate)

        fbank = np.zeros((numOfMelFilters, int(np.floor(FFTSize / 2 + 1))))
        for m in range(1, numOfMelFilters + 1):
            f_m_minus = int(bin[m - 1])
            f_m = int(bin[m])
            f_m_plus = int(bin[m + 1])

            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
                
        filter_banks = np.dot(pow_frames, fbank.T)
        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
        filter_banks = 20 * np.log10(filter_banks)
        
        # 6. DCT
        mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (numOfDctOutputs + 1)]
        
        # Ortalamasını alıp tek vektör yap (Neural Network girişi için)
        mfcc_feature = np.mean(mfcc, axis=0)
        
        feature_list.append(mfcc_feature)
        
        # Etiketi al (Dosya isminden)
        label = recording[1].split('_')[0]
        label_list.append(label)

    return np.array(feature_list), np.array(label_list)