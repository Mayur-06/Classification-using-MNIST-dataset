# MNIST Handwritten Digits Classification

This repository contains code for training and evaluating a simple logistic regression model to classify handwritten digits from the MNIST dataset. The MNIST dataset consists of 28x28 pixel grayscale images of handwritten digits (0 to 9), along with their corresponding labels.

## Requirements

- Python 3.x
- PyTorch
- torchvision
- matplotlib
- openpyxl
- pandas (for saving evaluation results to Excel)

You can install the required dependencies using pip:

```bash
pip install torch torchvision matplotlib pandas openpyxl
```

## Getting Started

1. Clone the repository:

```bash
git clone https://github.com/your_username/mnist-digit-classification.git
```

2. Navigate to the repository directory:

```bash
cd mnist-digit-classification
```

3. Run the provided Jupyter notebook (`mnist_classification.ipynb`) to train the model, evaluate its performance, and save the trained model's weights and evaluation logs.

## Training the Model

- Load the MNIST dataset using PyTorch's torchvision module.
- Preprocess the dataset, including transformations such as converting images to tensors and normalizing pixel values.
- Define and train a logistic regression model using PyTorch's nn module.
- Evaluate the trained model's performance on a validation dataset.
- Save the trained model's weights to a file (`mnist-logistic.pth`) for future use.
- Save the evaluation results (accuracy, loss) to a log file (`evaluation_logs.txt`).

## Evaluating the Model

- Load the trained model's weights from the saved file (`mnist-logistic.pth`).
- Evaluate the model's performance on a separate test dataset to assess its generalization ability.
- Calculate metrics such as accuracy and loss on the test dataset.
- Save the evaluation results to the log file (`evaluation_logs.txt`).

## Results

- Visualize the training and validation loss and accuracy over multiple epochs.
- Display sample predictions along with their true labels and model-predicted labels.

## Conclusion

The trained logistic regression model achieves a high accuracy on the MNIST test dataset, demonstrating its effectiveness in classifying handwritten digits. Additionally, the evaluation logs provide insights into the model's performance metrics during training and evaluation.
