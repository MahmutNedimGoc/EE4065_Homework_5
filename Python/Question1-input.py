import serial
import time
import numpy as np
import librosa
import struct

# --- AYARLAR ---
COM_PORT = 'COM3'     # Portunu kontrol et
BAUD_RATE = 115200    # Hızın neyse o (921600 öneririm ama 115200 de olur)
WAV_FILE = "our_sound.wav" # Test edeceğin dosya
BUFFER_SIZE = 6144    # 6 parça x 1024 örnek (STM32 bunu bekliyor)

def send_raw_audio():
    try:
        print(f"1. '{WAV_FILE}' dosyası okunuyor...")
        
        # Sesi yükle (8000 Hz, Mono)
        audio, _ = librosa.load(WAV_FILE, sr=8000, mono=True)
        
        # Normalize et (Ses seviyesini -1 ile +1 arasına çek)
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
            
        # BOYUT AYARLAMA (Tam 6144 örnek olmalı)
        if len(audio) < BUFFER_SIZE:
            # Kısa ise sonunu sıfırla doldur
            audio = np.pad(audio, (0, BUFFER_SIZE - len(audio)), 'constant')
        elif len(audio) > BUFFER_SIZE:
            # Uzun ise sadece en başını (veya ortasını) al
            audio = audio[:BUFFER_SIZE]
            
        # Float32 formatına çevir
        audio_data = audio.astype(np.float32)

        print(f"2. {COM_PORT} portuna bağlanılıyor...")
        ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=2)
        time.sleep(2) 
        
        # Byte dizisine çevir ve gönder
        payload = audio_data.tobytes()
        print(f"3. Ham Ses Verisi gönderiliyor ({len(payload)} byte)...")
        ser.write(payload)
        
        print(">>> GÖNDERİLDİ! Artık top STM32'de. Matematiği o yapacak.")
        ser.close()

    except Exception as e:
        print(f"HATA: {e}")

if __name__ == "__main__":
    send_raw_audio()