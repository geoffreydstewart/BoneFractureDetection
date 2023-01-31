#!/usr/bin/env python3


"""
This is an executable python3 script which can detect the state of a bone from an x-ray. It is currently capable of
detecting either a 'normal' state, or fracture state.
"""


import matplotlib.pyplot as plt
import numpy as np
from optparse import OptionParser
import os
from pathlib import Path
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


class ValidationException(Exception):
    """ Exception to indicate errors during command option validation. """
    def __init__(self, message=''):
        super(ValidationException, self).__init__(message)
        self.message = message


class ExecutionException(Exception):
    """ Exception to indicate errors during command option validation. """
    def __init__(self, message=''):
        super(ExecutionException, self).__init__(message)
        self.message = message


TRAIN_OP = "train"
PREDICT_OP = "detect"

VALID_OPERATIONS = [TRAIN_OP, PREDICT_OP]

# TODO: changes
IMG_HEIGHT = 200
# IMG_WIDTH = 75
IMG_WIDTH = 200

CHECKPOINT_FILENAMES = 'bonestatemodel.ckpt'
CHECKPOINT_DIRNAME = 'checkpoints'
CHECKPOINT_PATH = '%s/%s' % (CHECKPOINT_DIRNAME, CHECKPOINT_FILENAMES)


def train(options) -> None:
    # TODO: changes
    # batch_size = 8
    batch_size = 64

    data_dir = Path(options.input)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        # TODO: changes
        # color_mode='grayscale',
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=batch_size)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        # TODO: changes
        # color_mode='grayscale',
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=batch_size)

    class_names = train_ds.class_names
    print('This dataset contains the classes: %s' % class_names)

    # print the dimensions of the first instance of the training data
    for image_batch, labels_batch in train_ds:
        print(image_batch.shape)
        print(labels_batch.shape)
        break

    model = create_model()

    # Create a callback that saves the model's weights
    # cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=CHECKPOINT_PATH,
    #                                                  save_weights_only=True,
    #                                                  verbose=1)

    # TODO: changes
    # epochs = 25
    epochs = 4
    # history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[cp_callback])
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

    model.save_weights(CHECKPOINT_PATH)

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


def create_model() -> keras.Model:
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal",
                          input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.2),
        layers.RandomBrightness(0.4)
    ])

    model = Sequential([
        # TODO: changes
        # data_augmentation,
        # layers.Rescaling(1./255),
        layers.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    model.summary()
    return model


def detect(options) -> None:
    if not os.path.exists(CHECKPOINT_DIRNAME):
        raise ExecutionException('The checkpoints directory %s was not found. There does not appear to be an existing '
                                 'trained model.' % CHECKPOINT_DIRNAME)

    checkpoints = list(Path(CHECKPOINT_DIRNAME).rglob('%s*' % CHECKPOINT_FILENAMES))
    if not checkpoints:
        raise ExecutionException('No checkpoint files were found under the %s directory.' % CHECKPOINT_DIRNAME)

    model = create_model()

    checkpoint_dir = os.path.dirname(CHECKPOINT_PATH)
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    model.load_weights(latest)

    print('Detecting images found at: %s' % options.input)
    for image_name in os.listdir(options.input):
        if image_name.startswith('.'):
            continue
        image_path = os.path.join(options.input, image_name)
        # TODO: changes
        # img = tf.keras.utils.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH), color_mode='grayscale')
        img = tf.keras.utils.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        prediction = model.predict(img_array)
        print(prediction)
        if prediction[0][0] < 0.5:
            print('%s is Cat' % image_path)
        else:
            print('%s is a Dog' % image_path)


def validate_input():
    """
    Validates script inputs
    :rtype: options - the provided options and the operation argument
    :raises: ValidationException when there is an error
    """
    usage = "Usage: %prog [options]"
    # Since we are using Python3, it actually might be better to use argparse instead of optparse
    parser = OptionParser(usage=usage)
    parser.add_option("-i", "--input", dest="input", default="", help="Input Images Directory")

    options, args = parser.parse_args()
    operation = get_operation(args)
    if operation not in VALID_OPERATIONS:
        raise ValidationException("%s is not a valid operation. Try one of %s" % (operation, VALID_OPERATIONS))

    if not options.input:
        raise ValidationException('The option "i" must be provided such as -i /path/to/images')
    return options, operation


def get_operation(args):
    if len(args) >= 1:
        operation = args[0]
    else:
        raise ValidationException("At least one argument, a valid operation: One of %s must be specified." % VALID_OPERATIONS)
    return operation


def main():
    try:
        options, operation = validate_input()
        print("Performing OPERATION: %s with options %s" % (operation, options))
        if operation == TRAIN_OP:
            train(options)
        # for now, only other possibility here is detect
        else:
            detect(options)
    except (ValidationException, ExecutionException) as e:
        print(e.message)


if __name__ == "__main__":
    main()

