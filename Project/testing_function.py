import numpy as np
import tensorflow as tf
import cv2
import matplotlib
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
model = load_model('saved_model')


def get_prediction(img):

    # checking is the images is database image of real life image.
    if type(img) is str:
        img = cv2.imread(img)
        # converting the image to gray scale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)

    newimg = tf.keras.utils.normalize(resized)  # normalizing the data.

    # kernal operation for convolution layer.
    newimg = np.array(newimg).reshape(-1, 28, 28, 1)

    predicions = model.predict(newimg)  # pridicting the digit.

    return plt.imshow(img), plt.title("Original Image"), plt.show(), print('Digit in the image[according to model] is:', np.argmax(predicions))
