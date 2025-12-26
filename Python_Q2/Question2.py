import os
import numpy as np
import cv2
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
#from mnist import load_images, load_labels #Dont Work
import struct
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from matplotlib import pyplot as plt

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


train_img_path = os.path.join("MNIST-dataset", "train-images.idx3-ubyte")
train_label_path = os.path.join("MNIST-dataset", "train-labels.idx1-ubyte")
test_img_path = os.path.join("MNIST-dataset", "t10k-images.idx3-ubyte")
test_label_path = os.path.join("MNIST-dataset", "t10k-labels.idx1-ubyte")


train_images = load_images(train_img_path)
train_labels = load_labels(train_label_path)
test_images = load_images(test_img_path)
test_labels = load_labels(test_label_path)


train_huMoments = np.empty((len(train_images), 7))
test_huMoments = np.empty((len(test_images), 7))


for train_idx, train_img in enumerate(train_images):
    train_moments = cv2.moments(train_img, True)
    train_huMoments[train_idx] = cv2.HuMoments(train_moments).reshape(7)

for test_idx, test_img in enumerate(test_images):
    test_moments = cv2.moments(test_img, True)
    test_huMoments[test_idx] = cv2.HuMoments(test_moments).reshape(7)

model = keras.models.Sequential([
    keras.layers.Dense(100, input_shape=[7], activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

categories = np.unique(test_labels)

model.compile(loss=keras.losses.SparseCategoricalCrossentropy(),
              optimizer=keras.optimizers.Adam(1e-4))

mc_callback = ModelCheckpoint("mlp_mnist_model.h5")
es_callback = EarlyStopping(monitor="loss", patience=5)

model.fit(train_huMoments, train_labels, epochs=1000, verbose=1, 
          callbacks=[mc_callback, es_callback])

nn_preds = model.predict(test_huMoments)
predicted_classes = np.argmax(nn_preds, axis=1)

conf_matrix = confusion_matrix(test_labels, predicted_classes)
cm_display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=categories)

cm_display.plot()
cm_display.ax_.set_title("Neural Network Confusion Matrix")
plt.show()

header_content = """
#ifndef MODEL_DATA_MLP_H
#define MODEL_DATA_MLP_H

#include <stdint.h>


"""

def write_matrix(name, matrix):
    flat = matrix.flatten()
    c_code = f"const float {name}[{len(flat)}] = {{\n    "
    c_code += ", ".join([f"{x:.8f}f" for x in flat])
    c_code += "\n};\n\n"
    return c_code

w1, b1 = model.layers[0].get_weights()
header_content += f"// Layer 1: Weights ({w1.shape[0]}x{w1.shape[1]}) and Bias ({b1.shape[0]})\n"
header_content += write_matrix("W1", w1) 
header_content += write_matrix("B1", b1)

w2, b2 = model.layers[1].get_weights()
header_content += f"// Layer 2: Weights ({w2.shape[0]}x{w2.shape[1]}) and Bias ({b2.shape[0]})\n"
header_content += write_matrix("W2", w2)
header_content += write_matrix("B2", b2)

w3, b3 = model.layers[2].get_weights()
header_content += f"// Output Layer: Weights ({w3.shape[0]}x{w3.shape[1]}) and Bias ({b3.shape[0]})\n"
header_content += write_matrix("W3", w3)
header_content += write_matrix("B3", b3)

sample_img = test_images[0] 
header_content += "// Sample Image (28x28)\n"
header_content += "const uint8_t sample_image[28][28] = {\n"
for row in sample_img:
    header_content += "    {" + ", ".join(map(str, row)) + "},\n"
header_content += "};\n\n"

header_content += "#endif // MODEL_DATA_Q2_H\n"

with open("model_data_q2.h", "w", encoding="utf-8") as f:
    f.write(header_content)
