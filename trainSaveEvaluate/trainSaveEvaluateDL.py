# https://keras.io/examples/vision/image_classification_from_scratch/
# dataset https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

import os

image_size = (180, 180)
batch_size = 128
input_shape = image_size + (3,)
dir_name = "PetImages_small"


# Filter out corrupted images
def filter_corrupted_image():
    num_skipped = 0
    for folder_name in ("Cat", "Dog"):
        folder_path = os.path.join("kagglecatsanddogs_5340/"+dir_name, folder_name)
        for fname in os.listdir(folder_path):
            fpath = os.path.join(folder_path, fname)
            try:
                fobj = open(fpath, "rb")
                is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
            finally:
                fobj.close()
            if not is_jfif:
                num_skipped += 1
                os.remove(fpath)


def load_data_set():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        "kagglecatsanddogs_5340/"+dir_name,
        validation_split=0.2,
        subset="training",
        seed=1337,
        image_size=image_size,
        batch_size=batch_size,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        "kagglecatsanddogs_5340/"+dir_name,
        validation_split=0.2,
        subset="validation",
        seed=1337,
        image_size=image_size,
        batch_size=batch_size,
    )
    return train_ds, val_ds


def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    # Entry block
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)



def train(model, train_ds, val_ds, epochs=25):
    # callbacks = [
    #     keras.callbacks.ModelCheckpoint("save_at_{epoch}.keras"),
    # ]
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_ds,
        epochs=epochs,
        # callbacks=callbacks,
        validation_data=val_ds,
    )
    model.save("cat_vs_dog_model.h5")


def test(model):
    model.load_weights("cat_vs_dog_model.h5")
    img = keras.utils.load_img(
        "kagglecatsanddogs_5340/PetImages/Cat/6779.jpg", target_size=image_size
    )
    plt.imshow(img)

    img_array = keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis

    predictions = model.predict(img_array)
    score = float(predictions[0])
    print(f"This image is {100 * (1 - score):.2f}% cat and {100 * score:.2f}% dog.")


if __name__ == "__main__":
    # filter_corrupted_image()
    train_ds, val_ds = load_data_set()
    model = make_model(input_shape=input_shape, num_classes=2)
    train(model, train_ds, val_ds, epochs=10)
    test(model)
