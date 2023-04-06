# BoneFractureDetection
A deep learning project which studies the detection of bone fractures in humerus and femur x-rays, using a very small dataset. The code uses TensorFlow version 2.11.0. This is an ongoing machine learning project.

Initially it is shown that a basic Convolutional Neural Network is not effective at correctly detecting the state of a bone in an x-ray when a model is trained on a tiny dataset of x-ray images. It scores an accuracy of 65% measured using the average from k-Fold Cross Validation over 10 folds. See [the naive approach](bone_fracture_detector_basic.py)

Next, an approach which uses Transfer learning with fine-tuning, leveraging a DenseNet121 architecture and loading weights pretrained on ImageNet, is demonstrated. Even with the tiny dataset under study, a model which predicts the state of a humerus or femur bone, either normal or fracture, with accuracy of %80 is achieved. The average is computed using the average from k-Fold Cross Validation over 10 folds. See [the transfer learning approach](bone_fracture_detector_transfer.py)

Currently, the best results are obtained by adding an initial round of training on a portion of the MURA dataset, before fine-tuning the model with the target dataset. Perhaps exposing the model to some initial x-ray data assists it to learn x-ray specific features to compensate the small dataset. See [the development branch code](https://github.com/geoffreydstewart/BoneFractureDetection/blob/secondRoundFinetune/bone_fracture_detector_trnsfr_dev.py) for these changes. See below for a description of the MURA dataset.

To continue further, another avenue to explore to see if this tiny dataset can be used to establish a more effective model, a Generative adversarial network (GAN) will be used to supplement the existing dataset. Stay tuned!

The dataset is not provided here to avoid any potential copyright issues. If you're interested to get access to this custom dataset, please reach out.

### Usage
Both scripts implement a similar command line interface. For example, to evaluate the model which uses transfer learning with k-Fold Cross Validation, execute the following (note that there should exist Fracture and Normal directories containing the appropriate x-ray images, under the provided directory):
```
python3 bone_fracture_detector_transfer.py eval-w-kfold -i /path/to/training/dataset
```

To train the model which uses transfer learning, and persist the learned weights, execute the following (note that there should exist Fracture and Normal directories containing the appropriate x-ray images, under the provided directory):
```
python3 bone_fracture_detector_transfer.py train -i /path/to/training/dataset
```

To use the trained model which loads persisted weights learned through transfer learning, to make predictions on new x-rays, execute the following (note that the directory should directly contain one or more xray images):
```
python3 bone_fracture_detector_transfer.py detect -i /path/to/xrays/detect
```

### References
The following links have been useful references:
* https://keras.io/guides/transfer_learning/
* https://machinelearningmastery.com/evaluate-performance-deep-learning-models-keras/

### MURA
[MURA](https://stanfordmlgroup.github.io/competitions/mura) is a large dataset of bone x-rays from Stanford. For this study, only the humerus x-rays were used.

