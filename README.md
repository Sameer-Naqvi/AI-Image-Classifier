# AI Image Classification

This project is an AI-powered image classification tool that uses a convolutional neural network (CNN) to classify images from the CIFAR-10 dataset. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes, including planes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks. The neural network is trained to recognize and classify images into one of these categories.

## Features
**Convolutional Neural Network (CNN):** A deep learning model with two convolutional layers followed by fully connected layers.

**Training on CIFAR-10 Dataset:** The model is trained on the CIFAR-10 dataset, which contains a variety of labeled images.

**Real-Time Classification:** After training, the model can classify new images into one of the 10 classes.

**Training Visualization:** Loss is printed at the end of each training epoch to monitor the training progress.

## Prerequisites
Before running the code, you need to install the following dependencies:

**Python 3.x**

**NumPy**

**Pillow (PIL)**

**PyTorch**

**TorchVision**

**You can install the required Python packages using pip:** pip install numpy pillow torch torchvision

## Project Structure
**main.py:** The main script that trains the neural network and evaluates its performance on the CIFAR-10 dataset.

**data/:** Directory where the CIFAR-10 dataset will be downloaded and stored.

## How It Works
**Data Loading:** The CIFAR-10 dataset is downloaded and loaded into the program using torchvision.datasets. The dataset is split into training and test sets.

**Model Architecture:** The NeuralNet class defines a simple CNN architecture with two convolutional layers followed by three fully connected layers.

**Training:** The model is trained using the stochastic gradient descent (SGD) optimizer with a learning rate of 0.001 and momentum of 0.9. The loss is calculated using cross-entropy loss, and the training process runs for 30 epochs.

**Evaluation:** During training, the loss for each epoch is printed out. After training, the model can be used to classify images from the test set or new images provided by the user.

## How to Run
**Clone the repository:** git clone https://github.com/your-username/ai-image-classification.git
cd ai-image-classification

**Run the training script:** python main.py

The script will download the CIFAR-10 dataset (if not already downloaded), train the CNN model, and print the training loss for each epoch.

## Example Output
During training, you will see output similar to the following:

Training epoch 0...

Loss: 1.8987

Training epoch 1...

Loss: 1.6645

...

Training epoch 29...

Loss: 0.6348

After training, the model can be used to classify new images.
