import os
import cv2
import tensorflow as tf
import struct 
#from mnist import load_images, load_labels # Dont Work
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def load_images(path):
    with open(path, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(num, rows, cols)
    return images

def load_labels(path):
    with open(path, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)
    return labels

train_img_path = os.path.join( "MNIST-dataset", "train-images.idx3-ubyte")
train_label_path = os.path.join( "MNIST-dataset", "train-labels.idx1-ubyte")
test_img_path = os.path.join( "MNIST-dataset", "t10k-images.idx3-ubyte")
test_label_path = os.path.join( "MNIST-dataset", "t10k-labels.idx1-ubyte")

train_images = load_images(train_img_path)
train_labels = load_labels(train_label_path)
test_images = load_images(test_img_path)
test_labels = load_labels(test_label_path)

train_huMoments = np.empty((len(train_images),7))
test_huMoments = np.empty((len(test_images),7))

for train_idx, train_img in enumerate(train_images):
 train_moments = cv2.moments(train_img, True)
 train_huMoments[train_idx] = cv2.HuMoments(train_moments).reshape(7)

for test_idx, test_img in enumerate(test_images):
 test_moments = cv2.moments(test_img, True)
 test_huMoments[test_idx] = cv2.HuMoments(test_moments).reshape(7)

features_mean = np.mean(train_huMoments, axis = 0)
features_std = np.std(train_huMoments, axis = 0)
train_huMoments = (train_huMoments - features_mean) / features_std

test_huMoments = (test_huMoments - features_mean) / features_std
model = tf.keras.models.Sequential([
 tf.keras.layers.Dense(1, input_shape = [7], activation = 'sigmoid')
 ])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss=tf.keras.losses.BinaryCrossentropy(), 
              metrics=[tf.keras.metrics.BinaryAccuracy()])

train_labels[train_labels != 0] = 1
test_labels[test_labels != 0] = 1

model.fit(train_huMoments,
          train_labels,
          batch_size = 128,
          epochs=50,
          class_weight = {0:8, 1:1},
          verbose=1) 

perceptron_preds = model.predict(test_huMoments)

conf_matrix = confusion_matrix(test_labels, perceptron_preds > 0.5)
cm_display = ConfusionMatrixDisplay(confusion_matrix = conf_matrix)
cm_display.plot()
cm_display.ax_.set_title( "Single Neuron Classifier Confusion Matrix")
plt.show()
model.save( "mnist_single_neuron.h5")

print("Weights:", model.layers[0].get_weights()[0])
print("Bias:", model.layers[0].get_weights()[1])
print("Mean:", features_mean)
print("Std:", features_std)

image_index = 0 
test_img = test_images[image_index]
true_label = test_labels[image_index]


moments = cv2.moments(test_img, True)
hu_moments = cv2.HuMoments(moments).reshape(7)


plt.imshow(test_img, cmap='gray')
plt.title(f"Label: {true_label}")
plt.show()

header_content = """
#ifndef MODEL_DATA_H
#define MODEL_DATA_H

#include <stdint.h>


"""

weights = model.layers[0].get_weights()[0].flatten()
header_content += "// Trained Weights\n"
header_content += "const float MODEL_WEIGHTS[7] = {\n    "
header_content += ", ".join([f"{w:.8f}f" for w in weights])
header_content += "\n};\n\n"


bias = model.layers[0].get_weights()[1][0]
header_content += "// Trained Bias\n"
header_content += f"const float MODEL_BIAS = {bias:.8f}f;\n\n"


header_content += "// Normalization Mean\n"
header_content += "const float MODEL_MEAN[7] = {\n    "
header_content += ", ".join([f"{m:.8e}f" for m in features_mean])
header_content += "\n};\n\n"


header_content += "// Normalization Std\n"
header_content += "const float MODEL_STD[7] = {\n    "
header_content += ", ".join([f"{s:.8e}f" for s in features_std])
header_content += "\n};\n\n"

sample_img_for_h = test_images[0] 

header_content += "// Sample Image for Testing (28x28)\n"
header_content += "const uint8_t sample_image[28][28] = {\n"
for row in sample_img_for_h:
    header_content += "    {" + ", ".join(map(str, row)) + "},\n"
header_content += "};\n\n"

header_content += "#endif // MODEL_DATA_Q1_H\n"

output_filename = "model_data_q1.h"
with open(output_filename, "w", encoding="utf-8") as f:
    f.write(header_content)

print(f"\nBASARILI: '{output_filename}' dosyasi olusturuldu.")
