# Sign Language Classification with Multinomial Logistic Regression

Multinomial Logistic Regression implemented from scratch using NumPy for Sign Language MNIST image classification. The project includes softmax classification, cross-entropy loss, L2 regularization and gradient descent optimization without using machine learning frameworks.

## Features

- Multinomial Logistic Regression implemented from scratch
- Softmax output layer
- Cross-Entropy Loss
- L2 regularization
- Gradient descent optimization
- Sign Language MNIST classification
- Accuracy monitoring during training
- Confusion matrix evaluation
- Fully vectorized NumPy implementation

## Technologies

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-Learn

## Model Architecture

Input (784)
→ Linear(24)
→ Softmax

## Results

- Trained on the Sign Language MNIST dataset
- Classified 24 sign language gesture classes
- Implemented softmax classification from scratch
- Applied L2 regularization during training
- Evaluated model performance using classification accuracy
- Generated confusion matrix and class-wise performance analysis
- Monitored training loss and accuracy throughout training

## What I Learned

- Implementing Multinomial Logistic Regression from scratch
- Applying Softmax for multi-class classification
- Computing Cross-Entropy Loss
- Using L2 regularization to reduce overfitting
- Implementing gradient descent optimization
- Evaluating classification models using confusion matrices

## Run

Install dependencies:

```bash
pip install -r requirements.txt
```