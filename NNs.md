# Basic-Image-Classification-Using-Neural-Networks

Rhyme
#Task 1: Introduction

This is a small Basic Image Classification project with TensorFlow.

This graph describes the problem that we are trying to solve visually. We want to create and train a model that takes an image of a hand written digit as input and predicts the class of that digit, that is, it predicts the digit or it predicts the class of the input image.

-----
Prerequisites
```python
pip install tensorflow
```

![image](https://github.com/felixphan9/AI-Projects/assets/143317965/08af9d53-af3b-4973-b860-f484c7b66fce)

Hand Written Digits Classification

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
from matplotlib import pyplot as plt
%matplotlib inline

plt.imshow(x_train[0], cmap = 'binary')
plt.show()
```

![image](https://github.com/felixphan9/AI-Projects/assets/143317965/09e07024-2055-4346-b0dc-a4115e71ee83)



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
y_test_encoded = to_categorical(y_test)
```
Validated Shapes
```python
print('y_train_encoded shape: ' ,y_train_encoded.shape)
print('y_test_encoded shape: ' ,y_test_encoded.shape)
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
{0, 1, 2, 3, 9, 11, 14, 16, 18, 23, 24, 25, 26, 27, 30, 35, 36, 39, 43, 45, 46, 49, 55, 56, 64, 66, 70, 78, 80, 81, 82, 90, 93, 94, 107, 108, 114, 119, 126, 127, 130, 132, 133, 135, 136, 139, 148, 150, 154, 156, 160, 166, 170, 171, 172, 175, 182, 183, 186, 187, 190, 195, 198, 201, 205, 207, 212, 213, 219, 221, 225, 226, 229, 238, 240, 241, 242, 244, 247, 249, 250, 251, 252, 253, 255}

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

{-0.38589016215482896, 1.306921966983251, 1.17964285952926, 1.803310486053816, 1.6887592893452241, 2.8215433456857437, 2.719720059722551, 1.1923707702746593, 1.7396709323268205, 2.057868700961798, 2.3633385588513764, 2.096052433197995, 1.7651267538176187, 2.7960875241949457, 2.7451758812133495, 2.45243393406917, 0.02140298169794222, -0.22042732246464067, 1.2305545025108566, 0.2759611966059242, 2.210603629906587, 2.6560805059955555, 2.6051688630139593, -0.4240738943910262, 0.4668798577869107, 0.1486820891519332, 0.3905123933145161, 1.0905474843114664, -0.09314821501064967, 1.4851127174188385, 2.7579037919587486, 1.5360243604004349, 0.07231462467953861, -0.13133194724684696, 1.294194056237852, 0.03413089244334132, 1.3451056992194483, 2.274243183633583, -0.24588314395543887, 0.772349715676489, 0.75962180493109, 0.7214380726948927, 0.1995937321335296, -0.41134598364562713, 0.5687031437501034, 0.5941589652409017, 0.9378125553666773, 0.9505404661120763, 0.6068868759863008, 0.4159682148053143, -0.042236572029053274, 2.7706317027041476, 2.1342361654341926, 0.12322626766113501, -0.08042030426525057, 0.16140999989733232, 1.8924058612716097, 1.2560103240016547, 2.185147808415789, 0.6196147867316999, 1.943317504253206, -0.11860403650144787, -0.30952269768243434, 1.9942291472348024, -0.2840668761916362, 2.6306246845047574, 2.286971094378982, -0.19497150097384247, -0.39861807290022805, 0.2886891073513233, 1.7523988430722195, 2.3887943803421745, 2.681536327486354, 1.4596568959280403, 2.439706023323771, 2.7833596134495466, 2.490617666305367, -0.10587612575604877, 1.5614801818912332, 1.9051337720170087, 1.6123918248728295, 1.268738234747054, 1.9560454149986053, 2.6433525952501564, 1.026907930584471}



Task 6: Creating a Model
Creating the Model

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
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

Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
 dense (Dense)               (None, 128)               100480    
 dense_1 (Dense)             (None, 128)               16512     
 dense_2 (Dense)             (None, 10)                1290      

Total params: 118282 (462.04 KB)
Trainable params: 118282 (462.04 KB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________


Task 7: Training the Model
Training the Model

```python
model.fit(x_train_norm, y_train_encoded, epochs=3)
```

Epoch 1/3
1875/1875 [==============================] - 4s 2ms/step - loss: 0.3693 - accuracy: 0.8918

Epoch 2/3
1875/1875 [==============================] - 3s 2ms/step - loss: 0.1792 - accuracy: 0.9473

Epoch 3/3
1875/1875 [==============================] - 3s 2ms/step - loss: 0.1371 - accuracy: 0.9595

Evaluating the Model

```python
loss, accuracy = model.evaluate(x_test_norm, y_test_encoded)
print('Test set accuracy:', accuracy * 100)
```
Test set accuracy: 96.13999724388123

Task 8: Predictions
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
    plt.grid(False)
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



![image](https://github.com/felixphan9/AI-Projects/assets/143317965/1b2ec2cb-9145-456d-9a21-41dc9ab3e69d)
