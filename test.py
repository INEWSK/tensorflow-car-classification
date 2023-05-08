import os
import argparse
import numpy as np
import tensorflow as tf
import pandas as pd

# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from tensorflow.keras.applications import VGG16
# from tensorflow.keras.applications.vgg16 import preprocess_input
from keras import models
from keras.utils import load_img, img_to_array
from keras.applications.resnet import preprocess_input

# Define the command line arguments
parser = argparse.ArgumentParser(
    description="Testing a pre-trained TensorFlow model on a folder of images"
)
parser.add_argument(
    "--model_path", type=str, help="path to the pre-trained TensorFlow model"
)
parser.add_argument(
    "--image_folder_path", type=str, help="path to the folder with images to be tested"
)
parser.add_argument("--output_file_path", type=str, help="path to the output text file")
parser.add_argument(
    "--target_size", type=int, default=224, help="target size of the input image"
)
args = parser.parse_args()

# Load data set classes from given csv file
# df = pd.read_csv("./data/number_of_samples.csv")
# # Change DataFrame index start at 1
# df.index = df.index + 1
# class_names = df.pop("Name of class")

dataset_dir = "./data/train"
classes = []
for filename in os.listdir(dataset_dir):
    classes.append(filename)

# Load the pre-trained model
model = models.load_model(args.model_path)

# Get a list of all the images in the folder
image_paths = [
    os.path.join(args.image_folder_path, file)
    for file in os.listdir(args.image_folder_path)
]

# Create an empty list to store the results
results = []

# Loop through each image in the folder and apply the model
for image_path in image_paths:
    # Load the image and preprocess it
    image = load_img(image_path, target_size=(args.target_size, args.target_size))
    image = img_to_array(image)
    image = tf.expand_dims(image, axis=0)
    image = preprocess_input(image)

    # Make a prediction using the model
    prediction = model.predict(image)

    score = np.squeeze(prediction)
    result = str(
        "This image most likely belongs to {} with a {:.2f}% confidence."
    ).format(class_names[np.argmax(score) + 1], 100 * np.max(score))
    # Add the result to the list of results
    results.append(result)

# Save the results to a text file
with open(args.output_file_path, "w") as f:
    for result in results:
        f.write(str(result) + "\n")
print(f"Text file saved to {args.output_file_path}")
