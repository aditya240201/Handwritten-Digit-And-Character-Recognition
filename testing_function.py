import numpy as np
import tensorflow as tf
import cv2
import matplotlib              
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model  
model=load_model('saved_model')

def get_prediction(img):
    
    if type(img) is str:                                #checking is the images is database image of real life image.                     
        img=cv2.imread(img)
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)       #converting the image to gray scale
    else: 
        gray=img
        
    resized = cv2.resize(gray,(28,28),interpolation=cv2.INTER_AREA)

    newimg=tf.keras.utils.normalize(resized)            #normalizing the data.

    newimg=np.array(newimg).reshape(-1,28,28,1)         #kernal operation for convolution layer.      

    predicions=model.predict(newimg)                    #pridicting the digit.   
    
    return plt.imshow(img),plt.title("Original Image"),plt.show(),print('Digit in the image[according to model] is:',np.argmax(predicions))