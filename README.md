# EE4065 – Embedded Digital Image Processing
### Homework 5: End-to-End Embedded Machine Learning

---

## 📄 Project Description

This repository contains the implementation for **EE4065 – Homework 4**. The project focuses on deploying end-to-end Machine Learning models onto an **STM32 microcontroller** for both audio and image processing tasks, following the methodologies from the course textbook *Embedded Machine Learning with Microcontrollers*.

The project is divided into two main sections:
1.  **Audio Processing (Q1):** Keyword Spotting (0-9 Digits) using a custom "Pure C" DSP pipeline.
2.  **Image Processing (Q2):** Handwritten Digit Recognition (0-9) using **Hu Moments** and an **MLP Neural Network**.

---

## 1️⃣ Q1: Keyword Spotting (Audio Classification)
*Implementation of a spoken digit recognizer using raw audio signals.*

### 🧠 Methodology
* **Feature Extraction (Pure C):** Unlike standard libraries, we implemented a custom, library-free DSP pipeline in C:
    * **FFT:** Custom Cooley-Tukey algorithm for frequency analysis.
    * **MFCC:** Manual calculation of Mel-Frequency Cepstral Coefficients (Hamming Window, Mel-Filterbank, DCT Matrix).
* **Model:** A Neural Network trained offline to classify spoken digits based on extracted features.
* **Embedded Strategy:** The STM32 processes high-volume raw audio, extracts features on-chip, and performs inference via X-CUBE-AI.

---

## 2️⃣ Q2: Handwritten Digit Recognition (Image Classification)
*Multi-Class classification of MNIST digits (0-9) using Hu Invariant Moments.*

Instead of processing raw pixels, this implementation uses **Feature Extraction** to drastically reduce computational overhead and memory usage on the microcontroller.

### 🧠 Methodology
* **Feature Extraction (Hu Moments):**
    * Input images are processed to extract **7 Hu Invariant Moments**.
    * This reduces the input vector size from **784** to just **7**, making the model extremely lightweight.
* **Model Architecture (MLP):**
    A Multi-Layer Perceptron (MLP) was designed with the following topology:
    * **Input Layer:** 7 Neurons (Hu Moments).
    * **Hidden Layer 1:** 100 Neurons (Activation: **ReLU**).
    * **Hidden Layer 2:** 100 Neurons (Activation: **ReLU**).
    * **Output Layer:** 10 Neurons (Activation: **Softmax**) to classify digits 0-9.

### 💻 Embedded Implementation
The model was trained offline in Python (TensorFlow/Keras) and deployed to the STM32F4:
1.  **Weight Export:** Weights ($W$) and Biases ($b$) were exported to a C header file (`model_data_q2.h`).
2.  **Inference Engine:** A custom C function performs the forward propagation:
    * h = ReLU(W \cdot x + b)
    * y = Softmax(W_{out} \cdot h + b_{out})
3.  **Result:** The microcontroller calculates the probability distribution and selects the index with the highest value (Argmax) as the predicted digit.

### 📊 Results
The MLP model successfully recognizes all 10 digits. The confusion matrix below shows the classification performance on the test set:

*As verified in the debug session, the STM32 inference engine perfectly matches the Python model's predictions (e.g., correctly classifying inputs '7' and '2' with high confidence).*

---

## 🛠 Tools & Hardware
* **Hardware:** STM32 Nucleo Board (ARM Cortex-M4)
* **IDE:** STM32CubeIDE (with X-CUBE-AI)
* **Languages:** C (Embedded), Python (Training)
* **Libraries:** OpenCV (Feature Extraction), TensorFlow/Keras, Standard C Math (`math.h`)

---

## 👥 Group Members

| Student ID | Name | Role |
| :--- | :--- | :--- |
| **150721053** | **Mahmut Nedim Göç** | Q1, Q2, Reporting |
| **150722031** | **Emre Güner** | Q1, Q2, Reporting |

---

## 📅 Due Date
**January 2, 2026 – 23:59 PM**
