#!/usr/bin/env python3

"""
This is an executable python3 script which is capable of detecting the state of a humerus or femur bone from an
x-ray, either a 'normal' state, or fracture state. Transfer learning with fine-tuning is used leveraging a
DenseNet121 architecture loading weights pretrained on ImageNet. This approach proves to be reasonably effective on
the tiny dataset under study.
"""

import matplotlib.pyplot as plt
import numpy as np
from optparse import OptionParser, Values
import os
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


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


TRAIN_OP = 'train'
PREDICT_OP = 'detect'
MEASURE_WITH_KFOLD = 'eval-w-kfold'

VALID_OPERATIONS = [TRAIN_OP, PREDICT_OP, MEASURE_WITH_KFOLD]

IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 6

CHECKPOINT_FILENAMES = 'bonestatemodel.ckpt'
CHECKPOINT_DIRNAME = 'checkpoints_transfer'
CHECKPOINT_PATH = '%s/%s' % (CHECKPOINT_DIRNAME, CHECKPOINT_FILENAMES)


def train(options) -> None:

    data_dir = Path(options.input)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE)

    class_names = train_ds.class_names
    print('This dataset contains the classes: %s' % class_names)

    # print the dimensions of the first instance of the training data
    for image_batch, labels_batch in train_ds:
        print(image_batch.shape)
        print(labels_batch.shape)
        break

    model, base_model = create_model()

    training_epochs = 25

    # this trains the top layer added on top of the base model which currently has its layers frozen
    model.fit(train_ds, validation_data=val_ds, epochs=training_epochs)

    # fine tune the entire model
    # Unfreeze the base_model. Note that it keeps running in inference mode since we passed `training=False` when
    # calling it, as described in the create_model function. This means that the batchnorm layers will not update
    # their batch statistics. This prevents the batchnorm layers from undoing all the training done so far.
    base_model.trainable = True
    model.summary()

    # Note that a very low learning rate is needed at this step, because a much larger model is being trained now
    # since the base model's layers have been unfrozen, and a tiny dataset is being used. This avoids overfitting
    # by preventing large weight updates. The pretrained weights need to be re-adapted in a small, incremental way.
    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[keras.metrics.BinaryAccuracy()],
    )

    finetuning_epochs = 10
    history = model.fit(train_ds, epochs=finetuning_epochs, validation_data=val_ds)

    # Persist the weights to enable the separate detect operation
    model.save_weights(CHECKPOINT_PATH)

    acc = history.history['binary_accuracy']
    val_acc = history.history['val_binary_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(finetuning_epochs)

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


def create_model() -> (keras.Model, keras.Model):
    base_model = keras.applications.densenet.DenseNet121(
        weights='imagenet',  # Load weights pre-trained on ImageNet.
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        include_top=False)  # Do not include the ImageNet classifier at the top.

    base_model.trainable = False

    # data augmentation is needed for the tiny dataset being used
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal",
                          input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.2),
        layers.RandomBrightness(0.4)
    ])

    # Create new model on top
    inputs = keras.Input(shape=(150, 150, 3))
    # Apply the data augmentation
    x = data_augmentation(inputs)

    # performs the required preprocessing of the input to make is usable by the pre-trained DenseNet weights
    x = keras.applications.densenet.preprocess_input(x)
    # scale_layer = keras.layers.Rescaling(1./255)
    # x = scale_layer(x)

    # The base model contains batchnorm layers. We want to keep them in inference mode when we unfreeze the base
    # model for fine-tuning, so we make sure that the base_model is running in inference mode here
    x = base_model(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.2)(x)  # Regularize with dropout
    outputs = keras.layers.Dense(1)(x)
    model = keras.Model(inputs, outputs)

    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=[keras.metrics.BinaryAccuracy()])

    model.summary()
    # the model is needed by both the train and detect operations, and the base_model is needed for the
    # fine-tuning step in the training operation
    return model, base_model


def detect(options) -> None:
    if not os.path.exists(CHECKPOINT_DIRNAME):
        raise ExecutionException('The checkpoints directory %s was not found. There does not appear to be an existing '
                                 'trained model.' % CHECKPOINT_DIRNAME)

    checkpoints = list(Path(CHECKPOINT_DIRNAME).rglob('%s*' % CHECKPOINT_FILENAMES))
    if not checkpoints:
        raise ExecutionException('No checkpoint files were found under the %s directory.' % CHECKPOINT_DIRNAME)

    model, base_model = create_model()

    # identify the latest checkpoint and load the persisted weights
    checkpoint_dir = os.path.dirname(CHECKPOINT_PATH)
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    model.load_weights(latest)

    print('Detecting images found at: %s' % options.input)
    for image_name in os.listdir(options.input):
        if image_name.startswith('.'):
            continue
        image_path = os.path.join(options.input, image_name)
        # img = tf.keras.utils.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH), color_mode='grayscale')
        img = tf.keras.utils.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        prediction = model.predict(img_array)
        print(prediction)
        if prediction[0][0] < 0.5:
            print('%s is Fracture' % image_path)
        else:
            print('%s is Normal' % image_path)


def eval_with_kfold(options) -> None:
    def get_label(file_path) -> int:
        # Convert the path to a list of path components
        parts = file_path.parts
        # The second to last is the class-directory
        one_hot = parts[-2] == class_names
        # Integer encode the label
        return tf.argmax(one_hot)

    def decode_img(file_path) -> np.ndarray:
        # load the image
        img = keras.utils.load_img(file_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
        # return the image converted to a ndarray
        return keras.utils.img_to_array(img)

    # def decode_img(file_path) -> np.ndarray:
    #     # Load the raw data from the file as a string
    #     img = Image.open(file_path)
    #     # Resize the image to the desired size
    #     # img = img.resize((IMG_HEIGHT, IMG_WIDTH))
    #     img = np.array(img)
    #     img = np.resize(img, (IMG_HEIGHT, IMG_WIDTH, 3))
    #     img = img.astype('float32')
    #     return np.asarray(img)

    data_dir = Path(options.input)
    image_list = list(data_dir.glob('*/*.jpg'))
    image_count = len(image_list)
    print(image_count)

    class_names = np.array(sorted([item.name for item in data_dir.glob('*') if item.name != '.DS_Store']))
    print(class_names)

    ds_x = np.array([decode_img(x) for x in image_list])
    ds_y = np.array([get_label(x) for x in image_list])

    k_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    losses = []
    accuracies = []
    for train_inds, test_inds in k_fold.split(ds_x, ds_y):
        training_epochs = 25

        model, base_model = create_model()

        # this trains the top layer added on top of the base model which currently has its layers frozen
        model.fit(ds_x[train_inds], ds_y[train_inds], epochs=training_epochs, verbose=0)

        # fine tune the entire model
        # Unfreeze the base_model. Note that it keeps running in inference mode since we passed `training=False` when
        # calling it, as described in the create_model function. This means that the batchnorm layers will not update
        # their batch statistics. This prevents the batchnorm layers from undoing all the training done so far.
        base_model.trainable = True
        model.summary()

        # Note that a very low learning rate is needed at this step, because a much larger model is being trained now
        # since the base model's layers have been unfrozen, and a tiny dataset is being used. This avoids overfitting
        # by preventing large weight updates. The pretrained weights need to be re-adapted in a small, incremental way.
        model.compile(
            optimizer=keras.optimizers.Adam(1e-5),
            loss=keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=[keras.metrics.BinaryAccuracy()],
        )

        finetuning_epochs = 10
        model.fit(ds_x[train_inds], ds_y[train_inds], epochs=finetuning_epochs, verbose=0)
        metrics = model.evaluate(ds_x[test_inds], ds_y[test_inds], verbose=0)
        print()
        print("%s: %.2f" % (model.metrics_names[0], metrics[0]))
        print("%s: %.2f%%" % (model.metrics_names[1], metrics[1] * 100))
        print()
        losses.append(metrics[0])
        accuracies.append(metrics[1] * 100)

    print('Averages after k-Fold Cross Validation')
    print("Average Loss:  %.2f (+/- %.2f)" % (np.mean(losses), np.std(losses)))
    print("Average Accuracy:  %.2f%% (+/- %.2f%%)" % (np.mean(accuracies), np.std(accuracies)))


def validate_input() -> (Values, list[str]):
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


def get_operation(args) -> str:
    if len(args) >= 1:
        operation = args[0]
    else:
        raise ValidationException("At least one argument, a valid operation: One of %s must be specified."
                                  % VALID_OPERATIONS)
    return operation


def main() -> None:
    try:
        options, operation = validate_input()
        print("Performing OPERATION: %s with options %s" % (operation, options))
        if operation == TRAIN_OP:
            train(options)
        elif operation == PREDICT_OP:
            detect(options)
        # for now, only other possibility here is evaluate with kfold
        else:
            eval_with_kfold(options)

    except (ValidationException, ExecutionException) as e:
        print(e.message)


if __name__ == "__main__":
    main()
