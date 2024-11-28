# GNN and CNN MNIST Classification

This repository implements and compares **Graph Neural Networks (GNNs)** and **Convolutional Neural Networks (CNNs)** for classifying handwritten digits from the MNIST dataset.

---

## 📖 Description

### 1. **GNN Model**
- **Paper Reference**: [A Graph Neural Network for Superpixel Image Classification](https://arxiv.org/abs/1905.00115) by Jianwu Long, Zeran Yan, and Hongfa Chen.
- Converts MNIST images into **superpixel-based graphs** using the **SLIC algorithm**.
- Uses **Graph Attention Network (GAT)** layers to process graph-structured data.
- Aggregates features globally using **global mean pooling** for classification.

### 2. **CNN Model**
- Standard Convolutional Neural Network tailored for grid-structured image data.
- Consists of 4 convolutional layers and 3 fully connected layers.
- Serves as a baseline for performance comparison with GNNs.

---

## 🗂️ Dataset

- **MNIST Dataset**:
  - A dataset of handwritten digits (0-9) with 60,000 training and 10,000 test images.
- **Graph Transformation**:
  - Each MNIST image is segmented into 75 **superpixels** using the **SLIC algorithm**.
  - Each superpixel forms a node in the graph, with edges connecting adjacent superpixels.

---

## ⚙️ Implementation Details

### **1. GNN Model**
- **Architecture**:
  - 4 Graph Attention Network (GAT) layers.
  - Fully connected layers for final classification.
- **Training**:
  - Batch size: 64
  - Loss: Negative Log Likelihood Loss (NLLLoss)
  - Optimizer: Adam
  - Hyperparameter tuning for hidden channels and learning rate.

### **2. CNN Model**
- **Architecture**:
  - 4 Convolutional layers with ReLU activation.
  - Fully connected layers for classification.
- **Training**:
  - Batch size: 64
  - Loss: Negative Log Likelihood Loss (NLLLoss)
  - Optimizer: Adam
  - 5 epochs for training.

---


## 🔧 Setup and Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/gnn-cnn-mnist-classification.git
   cd gnn-cnn-mnist-classification
