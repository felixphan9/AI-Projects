# Object Localization with TensorFlow

This project will guide you to use TensorFlow to create a model and train a model to perform object localization


![image](https://github.com/felixphan9/AI-Projects/assets/143317965/e5a4442e-e798-435d-9d94-b0049ba81eab)






## Task 1: Download and Visualize Data
```python
!wget https://github.com/hfg-gmuend/openmoji/releases/latest/download/openmoji-72x72-color.zip
!mkdir emojis
!unzip -q openmoji-72x72-color.zip -d ./emojis
!pip install tensorflow==2.4
```
Import all the necessary Python libraries:
- TensorFlow: A free and open-source software library for machine learning.
- NumPy: A library for scientific computing with Python.
- Matplotlib: A library for creating graphs and charts in Python.
- OS: A library for interacting with the operating system.

```python
matplotlib inline

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

from PIL import Image, ImageDraw
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPool2D, BatchNormalization, Dropout

print('Using TensorFlow version', tf.__version__)
```
Create a dictonary of emojis, where each emoji is represented by a two-element tuple: the first element is the name of the emoji, and the second element is the file name of the emoji image

```python
emojis = {
    0: {'name': 'happy', 'file': '1F642.png'},
    1: {'name': 'laughing', 'file': '1F602.png'},
    2: {'name': 'skeptical', 'file': '1F928.png'},
    3: {'name': 'sad', 'file': '1F630.png'},
    4: {'name': 'cool', 'file': '1F60E.png'},
    5: {'name': 'whoa', 'file': '1F62F.png'},
    6: {'name': 'crying', 'file': '1F62D.png'},
    7: {'name': 'puking', 'file': '1F92E.png'},
    8: {'name': 'nervous', 'file': '1F62C.png'}
}

plt.figure(figsize=(9, 9))

for i, (j, e) in enumerate(emojis.items()):
    plt.subplot(3, 3, i + 1)
    plt.imshow(plt.imread(os.path.join('emojis', e['file'])))
    plt.xlabel(e['name'])
    plt.xticks([])
    plt.yticks([])
plt.show()
```


![image](https://github.com/felixphan9/AI-Projects/assets/143317965/5b2db688-03df-4b08-b531-3cf71e58e160)


## Task 2: Create Examples

Iterates over the elements of the `emojis` dictionary. 
For each element the code 
- Opens the image file for the corresponding emoji.
- Converts the image to the RGBA color space. 
- Loads the image data into memory.
- Creates a new image in the RGB color space with the same size as the original image.
- Pastes the original image into the new image, using the alpha channel of the original image as a mask.
- Updates the 'image' key in the emojis dictionary for the corresponding class ID with the new image.

```python
for class_id, values in emojis.items():
    png_file = Image.open(os.path.join('emojis', values['file'])).convert('RGBA')
    png_file.load()
    new_file = Image.new("RGB", png_file.size, (255, 255, 255))
    new_file.paste(png_file, mask=png_file.split()[3])
    emojis[class_id]['image'] = new_file
```
`emojis`
{0: {'name': 'happy',
    'file': '1F642.png',
    'image': <PIL.Image.Image image mode=RGB size=72x72>},
 1: {'name': 'laughing',
    'file': '1F602.png',
    'image': <PIL.Image.Image image mode=RGB size=72x72>},
 2: {'name': 'skeptical',
    'file': '1F928.png',
    'image': <PIL.Image.Image image mode=RGB size=72x72>},
 3: {'name': 'sad',
    'file': '1F630.png',
    'image': <PIL.Image.Image image mode=RGB size=72x72>},
 4: {'name': 'cool',
    'file': '1F60E.png',
    'image': <PIL.Image.Image image mode=RGB size=72x72>},
 5: {'name': 'whoa',
    'file': '1F62F.png',
    'image': <PIL.Image.Image image mode=RGB size=72x72>},
 6: {'name': 'crying',
    'file': '1F62D.png',
    'image': <PIL.Image.Image image mode=RGB size=72x72>},
 7: {'name': 'puking',
    'file': '1F92E.png',
    'image': <PIL.Image.Image image mode=RGB size=72x72>},
 8: {'name': 'nervous',
    'file': '1F62C.png',
    'image': <PIL.Image.Image image mode=RGB size=72x72>}}

Function `create_example()` take no arguments and returns a tuple of 4 values:
- The image of an emoji
- The class ID of the emoji
- The row index of the emoji in the image.
- The column index of the emoji in the image.
How the function works:
- Generates a random class ID between 0 and 9.
- Creates a blank image with the size 144x144x3.
- Generates a random row index and column index between 0 and 72.
- Pastes the emoji image corresponding to the random class ID into the blank image at the specified row and column indices.
- Scales the image to the uint8 format and returns it along with the class ID, row, and column indices.

```python
def create_example():
  class_id = np.random.randint(0, 9)
  image = np.ones((144, 144, 3)) * 255 
  row = np.random.randint(0, 72)       
  col = np.random.randint(0, 72)
  image[row: row + 72, col: col + 72, :] = np.array(emojis[class_id]['image'])
  return image.astype('uint8'), class_id, (row + 10) / 144, (col + 10) / 144
```

```python
image, class_id, row, col = create_example()
plt.imshow(image)
```

![image](https://github.com/felixphan9/AI-Projects/assets/143317965/6567bbc9-1edb-4858-ae21-88ea13db5de0)

## Task 5: Data Generator

Function `data_generator()` will generates batches of data for machine learning model.
The function takes a batch size as input and returns a tuple of two dictionaries:
1. The first dictionary contains the images in the batch.
2. The second dictionary contains the class labels and bounding boxes for the images in the batch.

```python
def data_generator(batch_size=16):
  while True:
    x_batch = np.zeros((batch_size, 144, 144, 3))
    y_batch = np.zeros((batch_size, 9))
    bbox_batch = np.zeros((batch_size, 2))

    for i in range(0, batch_size):
      image, class_id, row, col = create_example()
      x_batch[i] = image /255.
      y_batch[i, class_id] = 1.0
      bbox_batch[i] = np.array([row, col])
    yield {'image': x_batch}, {'class_out': y_batch, 'box_out': bbox_batch}
```
Using the `data_generator()` to generates batches of data. Extracting the image and class label from the batch. The class label is the index of the emoji in the emojis dictionary. The bounding box is a two-element tuple that specifies the row and column indices of the emoji in the image.
Then plot the bounding box on the image.

```python
example, label = next (data_generator (1))
image = example ['image'][0]
class_id = np.argmax (label['class_out'][0])
coords = label ['box_out'][0]

image = plot_bounding_box (image, coords, norm=True)
plt.imshow(image)
plt.title(emojis[class_id]['name'])
plt.show
```


![image](https://github.com/felixphan9/AI-Projects/assets/143317965/41afe731-af46-4247-ba8b-161927bf57d9)

## Task 6: Model

Create an CNN for classifying emojis. The model has the following architecture
An input layer with shape (144, 144, 3)
 - 5 convolutional layers with 2**(4 + i) filters of size 3 and ReLU activation for i = 0, 1, 2, 3, 4.
So the number of filters in the convolutional layer will be 24, 25, 26, 27 and 2**8 for i = 0, 1 ,2 ,3 and 4
- 2 batch normalization layers
- 4 max pooling layers of size 2
- A flatten layer
- A dense layer with 256 neurons and ReLU activation
- A dense layer with 9 neurons and softmax activation for the class output
- A dense layer with 2 neurons for the bounding box output

```python
input_ = Input (shape=(144, 144, 3), name='image')

x = input_

for i in range(0, 5):
  n_filters = 2**(4 + i)
  x = Conv2D(n_filters, 3, activation='relu')(x)
  x = BatchNormalization () (x)
  x = MaxPool2D (2) (x)

x = Flatten() (x)
x = Dense (256, activation='relu') (x)

class_out = Dense (9, activation='softmax', name='class_out') (x)
box_out = Dense (2, name='box_out') (x)

model= tf.keras.models. Model (input_, [class_out, box_out])
model.summary()
```
Model: "model":
Total params: 659,819
Trainable params: 658,827
Non-trainable params: 992

## Task 7: Custom Metric: IoU

Defines a custom metric called IoU (Intersection over Union), the IoU metric is used to measure the overlap between the predicted bounding box and the ground truth bounding box. The metric takes two arguments: the ground truth bounding box and the predicted bounding box.
- The ground truth bounding box: A 2-dimensional array that contains the row and column indices of the top-left corner and the bottom-right corner of the bounding box.
- The predicted bounding box is also a 2-dimensional array that contains the row and column indices of the top-left corner and the bottom-right corner of the bounding box.

Calculates the intersection area and the union area of the 2 bounding boxes.
The IoU metric is then calculated by dividing the intersection area by the union area.

```python
class IoU(tf.keras.metrics.Metric):
  def __init__(self, **kwargs):
    super(IoU, self).__init__(**kwargs)

    self.iou = self.add_weight(name='iou', initializer='zeros')
    self.total_iou = self.add_weight(name='total_iou', initializer='zeros')
    self.num_ex = self.add_weight(name='num_ex', initializer='zeros')

  def update_state(self, y_true, y_pred, sample_weight = None):
    def get_box(y):
      rows, cols = y_true[:, 0], y_true[:, 1]
      rows, cols = rows * 144, cols * 144
      y1, y2 = rows, rows + 52
      x1, x2 = cols, cols + 52
      return x1, y1, x2, y2

    def get_area(x1, y1, x2, y2):
      return tf.math.abs(x2 - x1) * tf.math.abs(y2 - y1)

    gt_x1, gt_y1, gt_x2, gt_y2 = get_box(y_true)
    p_x1, p_y1, p_x2, p_y2 = get_box(y_pred)

    i_x1 = tf.maximum(gt_x1, p_x1)
    i_y1 = tf.maximum(gt_y1, p_y1)
    i_x2 = tf.minimum(gt_x2, p_x2)
    i_y2 = tf.minimum(gt_y2, p_y2)

    i_area = get_area(i_x1, i_y1, i_x2, i_y2)
    u_area = get_area(gt_x1, gt_y1, gt_x2, gt_y2) + get_area(p_x1, p_y1, p_x2, p_y2) - i_area

    iou = tf.math.divide(i_area, u_area)
    self.num_ex.assign_add (1)
    self.total_iou.assign_add(tf.reduce_mean(iou))
    self.iou = tf.math.divide (self.total_iou, self.num_ex)

  def result(self):
    return self.iou

  def reset_state(self):
    self.iou = self.add_weight(name='iou', initializer='zeros')
    self.total_iou = self.add_weight (name='total_iou', initializer='zeros')
    self.num_ex = self.add_weight(name='num_ex', initializer='zeros')
```

## Task 8: Compile the Model
Compiles the emoji detection model. The model is compiles using the Adam optimizer and the categorical cross-entropy loss function for the class label output. The model is also compiled using the mean squared error (MSE) loss function for the bounding box output.

The model is also compiled with two metrics: accuracy and IoU. Accuracy is the percentage of examples that the model classifies correctly. IoU is the intersection over union metric, which measures the overlap between the predicted bounding box and the ground truth bounding box.

```python
model.compile(
    loss={
        'class_out': 'categorical_crossentropy',
        'box_out': 'mse'
    },
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    metrics={
        'class_out': 'accuracy',
        'box_out': IoU(name='iou')
    }
)
```

## Task 9: Custom Callback: Model Testing
Function `test_model()` takes two arguments: the emoji detection model and the test data generator. The function first gets the next batch of data from the test data generator. The function then predicts the class label and bounding box for the batch of data using the model.

The function then plots the ground truth bounding box and the predicted bounding box on the image. The function also displays the predicted class label and the ground truth class label on the image.

```python
def test_model(model, test_datagen):
  example, label = next(test_datagen)
  x = example['image']
  y = label['class_out']
  box = label['box_out']

  pred_y, pred_box = model.predict(x)

  pred_coords = pred_box[0]
  gt_coords = box[0]
  pred_class = np.argmax(pred_y[0])
  image = x[0]

  gt = emojis[np.argmax(y[0])]['name']
  pred_class_name = emojis[pred_class]['name']

  image = plot_bounding_box(image, gt_coords, pred_coords,norm=True)

  color = 'green' if gt == pred_class_name else 'red'

  plt.imshow(image)
  plt.xlabel(f'Pred: {pred_class_name}', color=color)
  plt.ylabel(f'GT: {gt}', color=color)
  plt.xticks([])
  plt.yticks([])
```
Test the model with one batch of images

```python
def test(model):
  test_datagen = data_generator(1)

  plt.figure(figsize=(16, 4))

  for i in range(0, 6):
    plt.subplot(1, 6, i + 1)
    test_model(model, test_datagen)
  plt.show()
```

![image](https://github.com/felixphan9/AI-Projects/assets/143317965/81ec2c1c-9ac5-4fec-a6b2-bb4916ffcfff)

The class `ShowTestImages`, an inheritance of `tf.keras.callbacks.Callback` class. The ShowTestImages class has one method called `on_epoch_end()`. The `on_epoch_end()` method is called at the end of each epoch. The method calls the test() function to visualize the predicted bounding boxes and class labels for a few images from the test set.
```python
class ShowTestImages(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    test(self.model)
```

## Task 10: Model Training

The function `lr_schedule()` takes two arguments: the current epoch number and the current learning rate. The function returns the learning rate for the next epoch.

```python
def lr_schedule(epoch, lr):
  if (epoch + 1) % 5 == 0:
    lr *= 0.2
  return max(lr, 3e-7)


_ = model.fit(
    data_generator(),
    epochs = 50,
    steps_per_epoch = 500,
    callbacks=[
               ShowTestImages(),
               tf.keras.callbacks.EarlyStopping(monitor='box_out_iou', patience= 3, mode = 'max'),
               tf.keras.callbacks.LearningRateScheduler(lr_schedule)
    ]
)
```
Epoch 1/50
![image](https://github.com/felixphan9/AI-Projects/assets/143317965/624e1948-a0fa-45a1-ab9e-d25e857150de)
20s 26ms/step - loss: 0.9364 - class_out_loss: 0.6752 - box_out_loss: 0.2612 - class_out_accuracy: 0.7770 - box_out_iou: 1.0000 - lr: 0.0010
Epoch 2/50
![image](https://github.com/felixphan9/AI-Projects/assets/143317965/b3ae9344-1015-4768-bcf0-df4aa8c4ce47)
11s 21ms/step - loss: 0.0378 - class_out_loss: 0.0098 - box_out_loss: 0.0280 - class_out_accuracy: 0.9995 - box_out_iou: 0.0000e+00 - lr: 0.0010
Epoch 3/50
![image](https://github.com/felixphan9/AI-Projects/assets/143317965/a1557c00-4bd2-4972-9aca-b7af525ade6c)
11s 22ms/step - loss: 0.0142 - class_out_loss: 0.0025 - box_out_loss: 0.0117 - class_out_accuracy: 1.0000 - box_out_iou: 0.0000e+00 - lr: 0.0010
Epoch 4/50
![image](https://github.com/felixphan9/AI-Projects/assets/143317965/0fe82389-74a3-4523-9465-94414404077b)
12s 23ms/step - loss: 0.0094 - class_out_loss: 0.0019 - box_out_loss: 0.0075 - class_out_accuracy: 1.0000 - box_out_iou: 0.0000e+00 - lr: 0.0010
