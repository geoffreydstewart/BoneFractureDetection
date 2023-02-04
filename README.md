# BoneFractureDetection
A deep learning project which studies the detection of bone fractures in humerus and femur x-rays, using a tiny dataset. The code uses TensorFlow version 2.11.0. 

Initially it is shown that a basic Convolutional Neural Network is not effective at correctly detecting the state of a bone in an x-ray when a model is trained on a tiny dataset of x-ray images. See [the naive approach](bone_fracture_detector_basic.py)

Next, an approach which uses Transfer learning with fine-tuning, leveraging a DenseNet121 architecture and loading weights pretrained on ImageNet, is demonstrated. Even with the tiny dataset under study, a model which predicts the state of a humerus or femur bone, either normal or fracture, with accuracy > 90% is achieved. See [the transfer learning approach](bone_fracture_detector_transfer.py)

The dataset is not provided here to avoid any potential copyright issues. If you're interested to get access to this dataset, please reach out.

### Usage
Both scripts implement a similar command line interface. For example, to train the model which uses transfer learning execute the following (note that there should exist Fracture and Normal directories containing the appropriate x-ray images, under the provided directory):
```
python3 bone_fracture_detector_transfer.py train -i /path/to/training/dataset
```

To use the trained model to make predictions on new x-rays, execute the following:
```
python3 bone_fracture_detector_transfer.py detect -i /path/to/dataset/predict
```
