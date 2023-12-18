[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/fEFF99tU)

# ml4science-synthetic-data
# Project 2: Machine Learning (CS-433)

Behavioural Data Synthesis for better fairness performances

**Team**: LosModelosMatemagicos <br>
**Team members**: [Yannick Detrois](https://github.com/YannickDetrois), [David Friou](https://github.com/AfroDeivid), [Adrien Vauthey](https://github.com/Lugiasc) 

# Table of contents
- [Project 2: Machine Learning (CS-433)](#project-2-machine-learning-cs-433)
- [Table of contents](#table-of-contents)
- [How to run our project](#how-to-run-our-project)
- [Folders and files](#folders-and-files)

- [Figures](#figures)

# How to run our project
The files to run can be found in the `src` directory. `script_oversample.py` is the main file of the project. Run it using this command when you are in the `src` directory:
```
python script_oversample.py --mode [baseline | labels | augmentation]
```
The `--mode` argument allows you to choose the method you want to use to oversample the data. 

# Folders and files
## `src/`
Contains the source code of the project

### `script_oversample.py`

### `bert_hyperparameter_tuning.py`
10-fold cross validation to find the best hyperparameters for the BERT model. Can be run to test hyperparameters one by one or by running a grid search.

## `ml/BERT/`
Contains our implementation of the BERT model

### `BERT.py`
Contains the functions that create the BERT model

### `BERTPipeline.py`
Contains the functions to train and predict with the BERT model such that it can be implemented in the main pipeline.

### `Config.py`
Configuration class for centralised management of simulation parameters.

### `MaskedLanguageModel.py`
A masked language model that extends the `tf.keras.Model` class.

### `MaskedTextGenerator.py`
Callback class for generating masked text predictions during training.

### `Vectorisation.py`
Manages the vectorisation encoding and decoding of state-action sequences.
Every state-action sequence is transformed into a unique token, taking care of special tokens (padding, breaks). Encode-decode using either a dictionary or a numpy array.

### `masking.py`
Generate masked input and corresponding labels for masked language modeling.

### `BertHyperparametersTuningTest.ipynb`
Notebook to test the implementation of the crossvalidation for BERT hyperparameters tuning.


# Figures
