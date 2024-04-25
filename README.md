# CIFAR-100 Multiclass Classifier

This repository contains a Keras-based implementation of a convolutional neural network (CNN) for image classification on the CIFAR-100 dataset.

## Dataset

The CIFAR-100 dataset is a collection of 60,000 32x32 color images in 100 classes, with 600 images per class. The dataset is divided into 50,000 training images and 10,000 test images.

## Model Architecture

The CNN model consists of multiple convolutional and max pooling layers, followed by flatten and dense layers. The model is trained using the Adam optimizer and categorical cross-entropy loss function.

## Features

- Implementation of a CNN model using Keras
- Training and testing on the CIFAR-100 dataset
- Data preprocessing and augmentation using Keras' ImageDataGenerator
- Visualization of training and validation accuracy and loss using matplotlib
- Prediction and visualization of test images using matplotlib

## Requirements

- Python 3.x
- Keras 2.x
- TensorFlow 2.x
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- OpenCV

## Usage

1. Clone the repository and navigate to the project directory.
2. Install the required packages using `pip install -r requirements.txt`.
3. Follow the instructions in the notebook to train and test the model.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! If you'd like to contribute to this project, please fork the repository and submit a pull request.
