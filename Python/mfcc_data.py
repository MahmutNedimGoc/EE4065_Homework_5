import numpy as np
from scipy.fftpack import dct
import math

# --- AYARLAR (Eğitiminle aynı olmalı) ---
SAMPLE_RATE = 8000
FFT_SIZE = 1024
NUM_MEL = 20
NUM_MFCC = 13
MIN_FREQ = 0
MAX_FREQ = SAMPLE_RATE / 2

def generate_header_file():
    filename = "mfcc_data.h"
    print(f"'{filename}' dosyası oluşturuluyor...")

    with open(filename, "w") as f:
        # Header Guard (Dosyanın 2 kere eklenmesini önler)
        f.write("#ifndef MFCC_DATA_H\n")
        f.write("#define MFCC_DATA_H\n\n")

        # 1. HAMMING PENCERESİ
        hamming = np.hamming(FFT_SIZE)
        f.write(f"// --- Hamming Window (Boyut: {FFT_SIZE}) ---\n")
        f.write(f"const float hamming_window[{FFT_SIZE}] = {{\n")
        f.write(", ".join([f"{x:.6f}f" for x in hamming]))
        f.write("\n};\n\n")

        # 2. MEL FILTRE BANKASI
        low_mel = 2595 * np.log10(1 + MIN_FREQ / 700.0)
        high_mel = 2595 * np.log10(1 + MAX_FREQ / 700.0)
        mel_points = np.linspace(low_mel, high_mel, NUM_MEL + 2)
        hz_points = 700 * (10**(mel_points / 2595.0) - 1)
        bin_points = np.floor((FFT_SIZE + 1) * hz_points / SAMPLE_RATE).astype(int)

        fbank = np.zeros((NUM_MEL, int(FFT_SIZE / 2 + 1)))
        for m in range(1, NUM_MEL + 1):
            f_m_minus = bin_points[m - 1]
            f_m = bin_points[m]
            f_m_plus = bin_points[m + 1]

            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bin_points[m - 1]) / (bin_points[m] - bin_points[m - 1])
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bin_points[m + 1] - k) / (bin_points[m + 1] - bin_points[m])

        flat_fbank = fbank.flatten()
        f.write(f"// --- Mel Filter Bank (Boyut: {NUM_MEL} x {int(FFT_SIZE/2 + 1)}) ---\n")
        f.write(f"const float mel_filters[{NUM_MEL} * {int(FFT_SIZE/2 + 1)}] = {{\n")
        f.write(", ".join([f"{x:.6f}f" for x in flat_fbank]))
        f.write("\n};\n\n")

        # 3. DCT MATRİSİ
        # Scipy DCT Type-2 Ortho Norm
        full_dct_matrix = dct(np.eye(NUM_MEL), type=2, norm='ortho', axis=0)
        # 1. indexten 13. indexe kadar (0 hariç)
        dct_matrix_truncated = full_dct_matrix[1:NUM_MFCC+1, :]
        flat_dct = dct_matrix_truncated.flatten()

        f.write(f"// --- DCT Matrix (Boyut: {NUM_MFCC} x {NUM_MEL}) ---\n")
        f.write(f"const float dct_matrix[{NUM_MFCC} * {NUM_MEL}] = {{\n")
        f.write(", ".join([f"{x:.6f}f" for x in flat_dct]))
        f.write("\n};\n\n")

        # Header Guard Kapatma
        f.write("#endif // MFCC_DATA_H\n")
    
    print("Tamamlandı! Dosya oluşturuldu.")

if __name__ == "__main__":
    generate_header_file()