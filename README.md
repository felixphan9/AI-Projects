# Basic-Image-Classification

Rhyme
#Task 1: Introduction

Welcome to Basic Image Classification with TensorFlow.

This graph describes the problem that we are trying to solve visually. We want to create and train a model that takes an image of a hand written digit as input and predicts the class of that digit, that is, it predicts the digit or it predicts the class of the input image.

Hand Written Digits Classification
Import TensorFlow

```
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
print('Using TensorFlow version', tf.__version__)
```

Task 2: The Dataset
Import MNIST

\u200b

Shapes of Imported Arrays

\u200b

Plot an Image Example

\u200b

Display Labels

\u200b

\u200b

Task 3: One Hot Encoding

After this encoding, every label will be converted to a list with 10 elements and the element at index to the corresponding class will be set to 1, rest will be set to 0:
original label 	one-hot encoded label
5 	[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
7 	[0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
1 	[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
Encoding Labels

\u200b

Validated Shapes

\u200b

Display Encoded Labels

\u200b

Task 4: Neural Networks
Linear Equations

Single Neuron

The above graph simply represents the equation:

\U0001d466=\U0001d4641\u2217\U0001d4651+\U0001d4642\u2217\U0001d4652+\U0001d4643\u2217\U0001d4653+\U0001d44f

Where the w1, w2, w3 are called the weights and b is an intercept term called bias. The equation can also be vectorised like this:

\U0001d466=\U0001d44a.\U0001d44b+\U0001d44f

Where X = [x1, x2, x3] and W = [w1, w2, w3].T. The .T means transpose. This is because we want the dot product to give us the result we want i.e. w1 * x1 + w2 * x2 + w3 * x3. This gives us the vectorised version of our linear equation.

A simple, linear approach to solving hand-written image classification problem - could it work?

Single Neuron with 784 features
Neural Networks

Neural Network with 2 hidden layers

This model is much more likely to solve the problem as it can learn more complex function mapping for the inputs and outputs in our dataset.
Task 5: Preprocessing the Examples
Unrolling N-dimensional Arrays to Vectors

\u200b

Display Pixel Values

\u200b

Data Normalization

\u200b

Display Normalized Pixel Values

\u200b

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

