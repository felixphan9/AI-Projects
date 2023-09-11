# Object Localization with TensorFlow




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
