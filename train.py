#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import tensorflow as tf

from keras import layers, applications
from keras import Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import image_dataset_from_directory, plot_model

BATCH_SIZE = 32
IMG_SIZE = (224, 224)

train_dir = "./data/train"
val_dir = "./data/val"

# loading dataset from directory
train_ds = image_dataset_from_directory(
    train_dir,
    labels="inferred",
    label_mode="int",
    interpolation="bilinear",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

val_ds = image_dataset_from_directory(
    val_dir,
    labels="inferred",
    label_mode="int",
    interpolation="bilinear",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

for x, y in train_ds.take(1):
    print(x.shape, y)

class_names = train_ds.class_names
print(class_names)
num_classes = len(class_names)

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")

# iterate over the training dataset
for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break

# configure the dataset for performance
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# standardize the data for neural network
normalization_layer = layers.Rescaling(1.0 / 255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]

# generating additional training data for small dataset to avoid over-fitting
data_augmentation = Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        # layers.RandomZoom(0.1),
    ]
)

# pre-trained model; https://keras.io/api/applications/
resNet_model = applications.ResNet50V2(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    classes=num_classes,
)

fine_tune_at = 100

for layer in resNet_model.layers[:fine_tune_at]:
    layer.trainable = False

model = Sequential(
    [
        layers.Rescaling(1.0 / 255, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        # data_augmentation,
        resNet_model,
        layers.GlobalAveragePooling2D(),  # REPLACE Flatten()
        # Fully Connected Layer
        # layers.Dense(512, activation="relu"),
        # layers.Dropout(0.5),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

# model compile
model.compile(
    optimizer=tf.optimizers.Nadam(learning_rate=1e-4),
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=["accuracy"],
)

# model summary
model.summary()

# plot model architecture
plot_model(model, show_shapes=False, rankdir="LR", show_layer_names=False, dpi=64)

# early stopping callback function
early_stop = EarlyStopping(monitor="val_loss", patience=7, verbose=1)

# reduce learning rate during model training
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=3,
    verbose=1,
    mode="min",
    min_lr=int(1e-5),
)

EPOCHS = 35

# model fitting
history = model.fit(
    train_ds,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=[early_stop, reduce_lr],
)

model.save("./model/")
del model

# data training result visualization
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]

epoch = range(1, len(acc) + 1)

loss = history.history["loss"]
val_loss = history.history["val_loss"]

plt.figure(figsize=(8, 8))

plt.subplot(1, 2, 1)
plt.plot(epoch, acc, label="Training Accuracy")
plt.plot(epoch, val_acc, label="Validation Accuracy")
plt.legend(loc="lower right")
plt.title("Training and Validation Accuracy")
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(epoch, loss, label="Training Loss")
plt.plot(epoch, val_loss, label="Validation Loss")
plt.legend(loc="upper right")
plt.title("Training and Validation Loss")
plt.grid()

plt.savefig("./acc_loss.png")
plt.show()
