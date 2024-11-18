#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 21:29:41 2024

@author: shahzaibhussain
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt


model = load_model("Shahzaib_model.h5")

# Function to preprocess a test image
def preprocess_image(image_path, target_size=(500, 500)):
    # Load the image
    img = load_img(image_path, target_size=target_size)
    # Convert the image to an array
    img_array = img_to_array(img)
    # Normalize the image (divide by 255)
    img_array = img_array / 255.0
    # Expand dimensions to match model input
    img_array = np.expand_dims(img_array, axis=0)
    return img_array, img

# Define the test images and their true labels
test_images = {
    "crack": {
        "path": "/Users/shahzaibhussain/Documents/AER 850/Project 2/Data/test/crack/test_crack.jpg",
        "label": "crack",
    },
    "missing_head": {
        "path": "/Users/shahzaibhussain/Documents/AER 850/Project 2/Data/test/missing-head/test_missinghead.jpg",
        "label": "missing-head",
    },
    "paint_off": {
        "path": "/Users/shahzaibhussain/Documents/AER 850/Project 2/Data/test/paint-off/test_paintoff.jpg",
        "label": "paint-off",
    },
}

# Function to visualize the predictions
def display_prediction(image, true_label, predicted_label, predictions, class_names):
    # Convert predictions to percentages
    probabilities = [f"{class_names[i]}: {predictions[0][i]*100:.1f}%" for i in range(len(class_names))]

    # Display the image
    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.axis('off')

    # Display the true and predicted labels
    plt.title(
        f"True Label: {true_label}\nPredicted Label: {predicted_label}",
        fontsize=14,
        color="black",
    )

    # Add the probabilities as text on the image
    for i, text in enumerate(probabilities):
        plt.text(
            10, 20 + i * 30, text, fontsize=12, color="green", backgroundcolor="white"
        )

    # Show the plot
    plt.show()

# Class names (adjust as per your model)
class_names = ["crack", "missing-head", "paint-off"]

# Predict and display results
for defect_type, details in test_images.items():
    # Preprocess the image
    processed_image, original_image = preprocess_image(details["path"])

    # Predict the class probabilities
    predictions = model.predict(processed_image)

    # Get the class with the highest probability
    predicted_class_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_class_index]

    # Display the result
    display_prediction(
        original_image,
        true_label=details["label"],
        predicted_label=predicted_class,
        predictions=predictions,
        class_names=class_names,
    )


