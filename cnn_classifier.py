"""CNN classifier for lithium-ion battery thermal images.

Public portfolio version of an experimental TensorFlow/Keras workflow.
The dataset loaders/generators are intentionally kept external so that
private research data are not distributed with this repository.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing import image


IMAGE_SIZE = (150, 150)
LEARNING_RATE = 5e-4


def build_model(input_shape=(150, 150, 3)):
    """Build and compile the binary CNN classifier."""
    model = Sequential(
        [
            Conv2D(
                32,
                (3, 3),
                activation="relu",
                input_shape=input_shape,
                kernel_regularizer=l2(0.01),
            ),
            MaxPooling2D((2, 2)),
            Dropout(0.30),
            Conv2D(64, (3, 3), activation="relu", kernel_regularizer=l2(0.01)),
            MaxPooling2D((2, 2)),
            Dropout(0.30),
            Conv2D(128, (3, 3), activation="relu", kernel_regularizer=l2(0.01)),
            MaxPooling2D((2, 2)),
            Dropout(0.50),
            Flatten(),
            Dense(256, activation="relu"),
            Dropout(0.50),
            Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model(model, train_generator, validation_generator, epochs=200):
    """Train the model using externally prepared Keras data generators."""
    return model.fit(
        train_generator,
        steps_per_epoch=9,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=3,
    )


def plot_history(history):
    """Plot training/validation accuracy and loss."""
    epochs = range(1, len(history.history["accuracy"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].plot(epochs, history.history["accuracy"], label="Training")
    axes[0].plot(epochs, history.history["val_accuracy"], label="Validation")
    axes[0].set_title("Accuracy")
    axes[0].legend()

    axes[1].plot(epochs, history.history["loss"], label="Training")
    axes[1].plot(epochs, history.history["val_loss"], label="Validation")
    axes[1].set_title("Loss")
    axes[1].legend()

    fig.tight_layout()
    plt.show()


def predict_image(model, image_path, threshold=0.5):
    """Return the sigmoid score for a single thermal image.

    Class-name mapping depends on the label encoding used when preparing the
    dataset, so this function intentionally returns the numeric score.
    """
    img = image.load_img(image_path, target_size=IMAGE_SIZE)
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0) / 255.0
    score = float(model.predict(img_tensor, verbose=0)[0, 0])
    return score, int(score >= threshold)


if __name__ == "__main__":
    model = build_model()
    model.summary()
