# BoneFractureDetection
A deep learning project which detects bone fractures in x-rays

The bone_state_detector.py program is designed to detect the state of a bone from an x-ray. It is currently capable of detecting either a 'normal' state, or fracture state.

First, check the imports to make sure all the dependencies are installed!

To train the model with a set of training images:
```
python3 bone_fracture_detector.py train -i /path/to/dataset/train
```

To test the model with a set of test images:
```
python3 bone_fracture_detector.py detect -i /path/to/dataset/test
```

Images used: Humerus, Femur x-rays