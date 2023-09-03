# Basic-Image-Classification-Using-CNNs

Rhyme
#Task 1: Introduction

Welcome to Basic Image Classification with TensorFlow using CNNs

This tutorial will perform basic image classification using CNNs on the CIFAR-10 dataset. 
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.

Here are the classes in the dataset, as well as 10 random images from each:

![image](https://github.com/felixphan9/Basic-Image-Classification/assets/143317965/597c974c-df35-4c2a-ad9d-87fd1a44f1a1)


Task 1: Introduction

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

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(128, activation='relu', input_shappe=(784,)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
```

Activation Functions

The first step in the node is the linear sum of the inputs:

![image](https://github.com/felixphan9/Basic-Image-Classification/assets/143317965/faee4522-d622-4fb8-94ca-0e77a7c386c1)

The second step in the node is the activation function output:

![image](https://github.com/felixphan9/Basic-Image-Classification/assets/143317965/09270f90-f6a6-4d40-92e5-47f3912f13fb)

Graphical representation of a node where the two operations are performed:

ReLU

![image](https://github.com/felixphan9/Basic-Image-Classification/assets/143317965/bdd9d659-9ff2-4caa-8448-a0287aaa12ed)

Compiling the Model

```python
model.compile(
  optimizer='sgd',
  loss='categorical_crossentropy',
  metrics=['accuracy']
)

model.summary()
```

Task 7: Training the Model
Training the Model

```python
model.fit(x_train_norm, y_train_encoded, epochs=3)
```

Evaluating the Model

```python
loss, accuracy = model.evaluate(x_test_norm, y_test_encoded)
print('Test set accuracy:', accuracy * 100)
```

Task 8: PredictionsÂ¶
Predictions on Test Set

```python
preds = model.predict(x_test_norm)
print('Shape of preds:', preds.shape)
```

Plotting the Results

```python
plt.figure(figsize=(12, 12))

start_index = 0

for i in range(25):
    plt.subplot(5, 5,i+1)
    plt.grids(False)
    plt.xticks([])
    plt.yticks([])

    pred = np.argmax(preds[start_index+i])
    gt = y_test[start_index+i]

    col = 'g'
    if pred != gt:
        col = 'r'
    plt.xlabel('i={}, pred={}, gt={}'.format(start_index+i, pred, gt), color=col)
    plt.imshow(x_test[start_index+i], cmap='binary')
plt.show()
```

