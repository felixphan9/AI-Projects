# Basic-Image-Classification

Rhyme
#Task 1: Introduction

Welcome to Basic Image Classification with TensorFlow.

This graph describes the problem that we are trying to solve visually. We want to create and train a model that takes an image of a hand written digit as input and predicts the class of that digit, that is, it predicts the digit or it predicts the class of the input image.

Hand Written Digits Classification
Import TensorFlow

```python
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
print('Using TensorFlow version', tf.__version__)
```

Task 2: The Dataset
Import MNIST
```python
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
```
Shapes of Imported Arrays
```python
print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print('x_test shape:', x_test.shape)
print('y_test shape:', y_test.shape)
```
Plot an Image Example
```python
from matplotlib import pypolt as plt
%matplotlib inline

plt.imshow(x_train[0], cmap = 'binary')
plt.show()
```

Task 3: One Hot Encoding

After this encoding, every label will be converted to a list with 10 elements and the element at index to the corresponding class will be set to 1, rest will be set to 0:
original label 	one-hot encoded label
5 	[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
7 	[0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
1 	[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
Encoding Labels
```python
from tensorflow.keras.utils import to_categorical
y_train_encoded = to_categorical(y_train)
y_test_encoded = to_categorical(y_test
```
Validated Shapes
```python
print('y_train_encoded shape: ' y_train_encoded.shape)
print('y_test_encoded shape: ' y_test_encoded.shape)
```

Task 4: Neural Networks
Linear Equations




![image](https://github.com/felixphan9/Basic-Image-Classification/assets/143317965/2da09245-87f7-439a-8b1e-76b4e0f5d34e)

Where the `w1, w2, w3` are called the weights and `b` is an intercept term called bias. The equation can also be *vectorised* like this:

Neural Network with 2 hidden layers





![image](https://github.com/felixphan9/Basic-Image-Classification/assets/143317965/f2804ac6-6e41-4ab4-96c3-d447fab88ddb)


  This model is much more likely to solve the problem as it can learn more complex function mapping for the inputs and outputs in our dataset.

Task 5: Preprocessing the Examples
Unrolling N-dimensional Arrays to Vectors
```python
import numpy as np

x_train_reshaped = np.reshape(x_train, (60000, 784))
x_test_reshaped = np.reshape(x_test, (10000, 784))

print('x_train_reshaped shape:', x_train_reshaped.shape)
print('x_test_reshaped shape:', x_test_reshaped.shape)
```
Display Pixel Values
```python
print(set(x_train_reshaped[0]))
```
Data Normalization
```python
x_mean = np.mean(x_train_reshaped)
x_std = np.std(x_train_reshaped)
epsilon = 1e-10

x_train_norm = (x_train_reshaped - x_mean) / (x_std + epsilon)
x_test_norm = (x_test_reshaped - x_mean) / (x_std + epsilon)
```
Display Normalized Pixel Values
```python
print(set(x_train_norm[0]))
```
Task 6: Creating a Model
Creating the Model

\u200b

Activation Functions

The first step in the node is the linear sum of the inputs:
\U0001d44d=\U0001d44a.\U0001d44b+\U0001d44f

The second step in the node is the activation function output:

\U0001d434=\U0001d453(\U0001d44d)

Graphical representation of a node where the two operations are performed:

ReLU
Compiling the Model

\u200b

Task 7: Training the Model
Training the Model

\u200b

Evaluating the Model

\u200b

Task 8: PredictionsÂ¶
Predictions on Test Set

\u200b

Plotting the Results

\u200b

\u200b

