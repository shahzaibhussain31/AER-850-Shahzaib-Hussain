#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 14:34:20 2024

@author: shahzaibhussain
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from keras_tuner import Hyperband
from tensorflow.keras.layers import LeakyReLU


#Step 1

# Define input image dimensions
img_width, img_height = 500, 500
batch_size = 3

# Define paths
train_dir = '/Users/shahzaibhussain/Documents/AER 850/Project 2/Data/train'
validation_dir = '/Users/shahzaibhussain/Documents/AER 850/Project 2/Data/valid' 

# Data augmentation for the training set (rescaling, shearing, zooming, and flipping)
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,       
    shear_range=0.2,         
    zoom_range=0.2,          
    horizontal_flip=True     
)

# Rescaling validation set
validation_datagen = ImageDataGenerator(rescale=1.0/255)

# Generate training data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

# Generate validation data
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)


#Step 2-3: Neural Network Architecture Design/Hyperparameter Analysis
#Initialize the model
model = Sequential()

#Convolutional Layer 1 with LeakyReLU
model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), input_shape=(500, 500, 3)))
model.add(LeakyReLU(alpha=0.1))  # LeakyReLU activation
model.add(MaxPooling2D(pool_size=(2, 2)))

#Convolutional Layer 2 with ELU
model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1)))
model.add(LeakyReLU(alpha=0.1))  # LeakyReLU activation
model.add(MaxPooling2D(pool_size=(2, 2)))

#Convolutional Layer 3 with ReLU
model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#Flatten layer
model.add(Flatten())

#Fully Connected Dense Layer with ELU and Dropout
model.add(Dense(units=128, activation='elu'))  # ELU activation
model.add(Dropout(0.5))

#Output layer with Softmax for multi-class classification
model.add(Dense(units=3, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#Summary of the model
model.summary()

#Step 4

history = model.fit(
    train_generator,
    epochs=10,  
    validation_data=validation_generator
)

# Plot Training and Validation Accuracy
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1) 
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot Training and Validation Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

model.save('Shahzaib_model.h5')



