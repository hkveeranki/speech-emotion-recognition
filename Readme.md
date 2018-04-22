## Speech Emotion Recognition
This repository contains our work on Speech emotion recognition using emodb dataset. This dataset is available here [Emo-db](http://www.emodb.bilderbar.info/download/)

### Prerequisites
Linux (preferable Ubuntu LTS). Python2.x 

### Installing dependencies 
Dependencies are listed below and in the `requirements.txt` file.

* h5py
* Keras
* scipy
* sklearn
* speechpy
* tensorflow
* tqdm

Install one of python package managers in your distro. If you install pip, then you can install the dependencies by running 
`pip2 install -r requirements.txt` 

If you prefer to accelerate keras training on GPU's you can install `tensorflow-gpu` by 
`pip2 install tensorflow-gpu`

### Details of the code

The code folder contains all the code required. It contains three files
- `utilities.py` - Contains code to read the files, extract the features and create test and train data
- `train_model.py` - Code to train non DL models. The code has three models
	- `SVM`
	- `Random Forest`
	- `Neural Network`
- `train_DNN.py` - Code to train Deep learning Models. Supports two models
    - `LSTM`
    - `CNN`
### Executing the code
Run `python2 train_<>.py` from the command line.
