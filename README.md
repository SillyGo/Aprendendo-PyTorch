# PyTorch Learning Experiments

This repository is where I keep PyTorch implementations for learning ML concepts through hands-on exmaples.

Most files are experiment snapshots from learning sessions. Some focus on basic architectures, some on specific tasks, and some explore different approaches to common problems.

## Repository Structure

```
Begginings/
    cifar10CNN.py
    LSTMpothole.py
    mlp.py
    mnistCNN.py
    potholeMLP.py
    regression.py
    torch_cnn1.py
    torchLSTM.py
```

## Method Families

### Classification Models

These are prototypes for image classification tasks, using conv or fully connected networks.

- Convolutional networks for CIFAR-10 and MNIST datasets.
- Multi-layer perceptrons for digit recognition.

### Regression and Time Series

Models for prediction tasks, including linear regression and sequence modeling.

- Linear regression on synthetic data.
- LSTM networks for time series forecasting and anomaly dectection.

### Specialized Applications

Experiments with real-world data, such as sensor-based pothole detection.

- MLP and LSTM variants using accelerometer and gyroscope data.

## File Overview

### `Begginings/cifar10CNN.py`
- Convolutional neural network for CIFAR-10 image classification.
- Pipeline summary:
  - Two convolutional layers with max pooling and dropout.
  - Fully connected layers with ReLU activation.
  - Trains with SGD optimizer and cross-entropy loss.
- Includes validation function for accuracy checking.

### `Begginings/LSTMpothole.py`
- LSTM-based model for pothole detection using accelerometer data.
- Pipeline summary:
  - Processes CSV sensor data and timestamps.
  - Creates sequences for time series prediction.
  - Note: This code is under development and may not fully work yet.

### `Begginings/mlp.py`
- Multi-layer perceptron for MNIST digit classification.
- Pipeline summary:
  - Single hidden layer with ReLU activation.
  - Uses cross-entropy loss and basic SGD.
- Demonstrates standard neural network training loop.

### `Begginings/mnistCNN.py`
- Convolutional neural network for MNIST digit recognition.
- Pipeline summary:
  - Two convolutional layers followed by fully connected layers.
  - Max pooling and ReLU activations.
- Similar to cifar10CNN but for grayscale images.

### `Begginings/potholeMLP.py`
- Multi-layer perceptron for pothole detection using gyroscope data.
- Pipeline summary:
  - Processes data from multiple trips.
  - Creates binary labels for pothole presence.
- Focuses on sensor data classification.

### `Begginings/regression.py`
- Linear regression model for synthetic data.
- Pipeline summary:
  - Generates noisy linear data and normalizes it.
  - Trains single linear layer with MSE loss.
- Includes prediction on new data.

### `Begginings/torch_cnn1.py`
- Enhanced convolutional neural network for CIFAR-10 with data augmentation.
- Pipeline summary:
  - Includes random flips, rotations, and color jittering.
  - Dropout for regularization.
- Uses transforms for data preprocessing.

### `Begginings/torchLSTM.py`
- LSTM network for sine wave time series prediction.
- Pipeline summary:
  - Generates sequences from sine data.
  - Trains LSTM to forecast next values.
- Demonstrates sequence creation and RNN training.

## Core PyTorch Concepts Used

The main PyTorch building blocks I use in these experiments are:

- `nn.Module`
  - Base class for defining custom neural networks.

- Layer types
  - `nn.Linear` for fully connected layers.
  - `nn.Conv2d` for convolutonal layers.
  - `nn.LSTM` for recurrent layers.
  - `nn.MaxPool2d` and `nn.Dropout` for regularization.

- Activation functions
  - `F.relu` for non-linear activations.

- Loss functions
  - `nn.CrossEntropyLoss` for classification.
  - `nn.MSELoss` for regression.

- Optimizers
  - `optim.SGD` and `optim.Adam` for parameter updates.

- Data handling
  - `torchvision.transforms` for image preprocessing.
  - `torch.utils.data.DataLoader` for batching.
  - `torchvision.datasets` for standard datasets.

- Training utilities
  - Manual training loops with forward/backward passes.
  - Device management with `torch.device`.

## Training Patterns

Most scripts follow similar training structures:

- Model definition inheriting from `nn.Module`.
- Data loading with transforms and DataLoader.
- Training loop with loss computation, backward pass, and optimizer step.
- Validation or testing for evaluation.

## Notes on Maturity

- These are learning experiments and snapshots, not optimized production code.
- Parameters are often hardcoded for simplicity.
- Some files may be incomplete or experimental.
- Computational requirements vary betwen models.

This README serves as a guide to the experiments. Images, videos, and performance notes will be added as the repository evolves.