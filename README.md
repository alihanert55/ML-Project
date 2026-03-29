# Human Action Recognition (HAR) via OpenPose and Custom Deep Learning Models

## 📌 Project Overview
This repository contains a comprehensive Machine Learning pipeline for **Video-Based Human Action Recognition (HAR)**. Unlike standard end-to-end 3D CNN approaches (which are computationally heavy), this project adopts a highly efficient, skeleton-based methodology. 

The core objective is to extract human skeletal keypoints from video frames using **OpenPose** and classify the underlying temporal actions using **Custom Deep Learning Architectures**. By translating raw video pixels into sequential spatial coordinates, the models achieve high accuracy while significantly reducing computational overhead.

## 🧠 Methodology & Custom Implementations

To handle the highly non-linear and variable nature of human movement, several custom preprocessing pipelines and network architectures were implemented from scratch.

### 1. Data Pipeline & Custom Preprocessing
Raw OpenPose outputs consist of $(x, y, c)$ coordinates (where $c$ is the confidence score) for multiple joints per frame. Since videos have arbitrary lengths and subjects appear at different scales, a custom data processing pipeline was built:

* **Custom Spatial Normalization (Translation & Scale Invariance):** Raw coordinates are highly dependent on the subject's position in the camera frame. A custom algorithm was implemented to shift the origin $(0,0)$ of every frame to a central reference joint (e.g., the Neck or Mid-Hip). Furthermore, the skeleton was scaled based on a reference distance (e.g., torso length) to ensure the model is invariant to camera distance.
* **Custom Feature Engineering (Kinematics):**
    Instead of relying solely on static coordinates, dynamic features were explicitly calculated. The pipeline computes the **velocity** (first derivative) and **acceleration** (second derivative) of the joints between consecutive frames, providing the network with explicit temporal motion cues.
* **Temporal Sequence Padding & Truncation:**
    Recurrent models require fixed-length sequence batches. A custom sequence generator was implemented to dynamically pad shorter video sequences with zeros (or last-frame repetition) and truncate longer sequences systematically, ensuring no critical action frames are lost.

### 2. Custom Model Architectures
To classify the time-series skeleton data, multiple model architectures were implemented and compared. Instead of using off-the-shelf wrappers, models were constructed using custom class definitions to allow fine-grained control over the forward pass.

* **Custom Recurrent Neural Networks (LSTM with Peephole Connections & GRU):**
    A sequence-to-vector architecture was implemented from scratch using Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) networks. Notably, the LSTM architecture was customized to incorporate **Peephole Connections**. Unlike standard library implementations that only use the input and previous hidden state for gating, this modification allows the gate layers (forget, input, and output) to directly observe the internal cell state. This enhancement significantly improves the network's ability to learn precise timings and subtle temporal dependencies in human kinematics. The custom forward pass evaluates these variable-length sequences, extracts the hidden state of the final unrolled time step, and routes it through a custom fully connected (Dense) classification head with Dropout for regularization.
* **1D Convolutional Neural Networks (1D-CNN):**
    A custom Temporal Convolutional Network approach was built. Here, the temporal frames are treated as the "spatial" dimension, and the flattened joint coordinates act as the input channels. The custom 1D-CNN layers extract local temporal patterns (e.g., a sudden arm wave) before applying Global Max Pooling and classification.
* **Multi-Layer Perceptron (MLP) Baseline:**
    A deep MLP was constructed as a baseline, utilizing flattened statistical features (mean, variance, max, min of joint movements across the video) to evaluate the necessity of sequential modeling.

## 📊 Training, Evaluation & Results

The models were trained utilizing custom training loops, allowing for explicit gradient clipping (to prevent exploding gradients in LSTMs) and dynamic learning rate scheduling.

* **Loss Function:** Categorical Cross-Entropy.
* **Optimization:** Adam optimizer with custom weight decay parameters.
* **Evaluation Metrics:** The models were evaluated not just on validation loss, but on precise test accuracy.

### Final Project Results
An automated evaluation script (`FINAL PROJECT RESULTS`) processes all trained models and generates comparative metrics. The final performance comparison is visualized using Seaborn bar plots, dynamically annotating the **Best Test Accuracy** for each custom implementation. This allows for a clear, data-driven conclusion on whether recurrent or convolutional architectures perform best for this specific skeleton-based HAR task.

## ⚙️ Requirements and Reproduction

### Dependencies
* `Python 3.x`
* `PyTorch` / `TensorFlow` (Depending on the specific notebook engine used)
* `NumPy`, `Pandas`
* `Matplotlib`, `Seaborn`
* `tslearn` (for time-series operations)
* Pre-extracted OpenPose `npz` datasets.

### How to Run
1. Ensure the pre-processed OpenPose coordinate dataset (`processed_dataset_npz.zip`) is present in your working directory or mounted via Google Drive.
2. Run the initialization cell to unzip the data and install requirements:
   ```bash
   pip install tslearn
   unzip processed_dataset_npz.zip
