import os
import numpy as np
import scipy.signal as sig
from mfcc_func import create_mfcc_features # Bu dosyanın (mfcc_func.py) yan klasörde olduğundan emin ol
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder

# Klasör ayarları
RECORDINGS_DIR = "recordings"

# Klasör kontrolü
if not os.path.exists(RECORDINGS_DIR):
    print(f"Hata: '{RECORDINGS_DIR}' klasörü bulunamadı.")
    recordings_list = []
else:
    recordings_list = [(RECORDINGS_DIR, recording_path) for recording_path in os.listdir(RECORDINGS_DIR)]

# Parametreler
FFTSize = 1024
sample_rate = 8000
numOfMelFilters = 20
numOfDctOutputs = 13
window = sig.get_window("hamming", FFTSize)

# Veri setini ayırma (Yweweler konuşmacısını test, diğerlerini train yapma)
test_list = {record for record in recordings_list if "yweweler" in record[1]}
train_list = set(recordings_list) - test_list

# MFCC Özelliklerini Çıkarma
print("Özellikler çıkarılıyor (Train)...")
train_mfcc_features, train_labels = create_mfcc_features(train_list, FFTSize, sample_rate, numOfMelFilters, numOfDctOutputs)
print("Özellikler çıkarılıyor (Test)...")
test_mfcc_features, test_labels = create_mfcc_features(test_list, FFTSize, sample_rate, numOfMelFilters, numOfDctOutputs)

# Input shape'i dinamik hale getirme
input_dim = train_mfcc_features.shape[1]

# Model Mimarisi
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(100, input_shape=(input_dim,), activation="relu"),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

# Label Encoding
ohe = OneHotEncoder()
train_labels_ohe = ohe.fit_transform(train_labels.reshape(-1, 1)).toarray()

# Test etiketlerini integer'a çevirme (Confusion Matrix için)
categories, test_labels_indices = np.unique(test_labels, return_inverse=True)

# Derleme
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), 
              optimizer=tf.keras.optimizers.Adam(1e-3), 
              metrics=['accuracy'])

# Eğitim
print("Eğitim başlıyor...")
model.fit(train_mfcc_features, train_labels_ohe, epochs=150, verbose=1)

# Tahmin
nn_preds = model.predict(test_mfcc_features)
predicted_classes = np.argmax(nn_preds, axis=1)

# Confusion Matrix Gösterimi
conf_matrix = confusion_matrix(test_labels_indices, predicted_classes)
cm_display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=categories)

cm_display.plot()
cm_display.ax_.set_title("Neural Network Confusion Matrix")
plt.show()

# 1. Modeli Normal Kaydet (.h5)
model.save("mlp_fsdd_model.h5")
print("\nModel .h5 olarak kaydedildi.")

# 2. TFLite'a Çevir ve Kaydet (.tflite)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print("Model .tflite formatına çevrildi ve 'model.tflite' olarak kaydedildi.")
print("ARTIK BU 'model.tflite' DOSYASINI STM32 X-CUBE-AI ARAYÜZÜNDEN SEÇEBİLİRSİN.")