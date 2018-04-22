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

### Directory Structure
- `code` - Contains all the code files
- `dataset` - Contains the speech files in wav formatted seperated into 7 folders which are the corresponding labels of those files
- `models` - Contains the saved models which obtained best accuracy on test data.

### Details of the code
- `utilities.py` - Contains code to read the files, extract the features and create test and train data
- `train_model.py` - Code to train non DL models. The code has three models with below given model numbers
	- `1 - SVM`
	- `2 - Random Forest`
	- `3 - Neural Network`
- `train_DNN.py` - Code to train Deep learning Models. Supports two models with below given model numbers
    - `1 - CNN`
    - `2 - LSTM`
### Executing the code
Run `python2 train_<>.py <model_number>` from the command line.
