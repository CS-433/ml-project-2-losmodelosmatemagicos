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

The configuration parameters are in the [`src/config.yaml`](https://github.com/CS-433/ml-project-2-losmodelosmatemagicos/blob/main/src/configs/config.yaml). You can change the parameters in this file according to your needs. \
Make sure to change the `root_name` according to what you are testing in the run. For example, if you make a simple run using the `baseline` mode, you should write `baseline` as the `root_name`.

When choosing `augmentation`, there are 4 different strategies (see [Fig.2](#fig1-original-distribution-of-the-data-and-the-4-oversampling-strategies)):
| Type | Description | Example |
| -- | ----------- | ------- |
| 1. | Balanced demographics with 50% original data and 50% synthetic data | [oo] [---] -> [oooOOO] [---...] |
| 2. | Balanced demographics with 100% synthetic data | [oo] [---] -> [OOO] [...] |
| 3. | Original demographics with 100% synthetic data | [oo] [---] -> [OO] [...] |
| 4. | Balanced demographics which are rebalanced with synthetic data | [oo] [---] -> [ooO] [---] |

o: sequence of demographic 1, O: SYNTHETIC sequence of demographic 1 \
-: sequence of demographic 2, .: SYNTHETIC sequence of demographic 2 

which can be selected in the [`config.yaml`](https://github.com/CS-433/ml-project-2-losmodelosmatemagicos/blob/main/src/configs/config.yaml) file by modifying the `type` value under `experiment` with values between 1-4 respectively (default to 1).

To change parameters for the BERT model, head in the [`Config.py`](https://github.com/CS-433/ml-project-2-losmodelosmatemagicos/blob/main/src/ml/BERT/Config.py) file. You can change the parameters in this file according to your needs.

## External libraries
The following libraries are required to run our project:
- `imbalanced-learn`
- `imblearn`
- `keras`
- `matplotlib`
- `numpy`
- `pandas`
- `pyyaml`
- `seaborn`
- `scikit-learn`
- `tensorflow`

Install them using the following command:
```
pip install -r requirements.txt
```

# Folders and files
List of all the files we implemented or modified for the scope of this project. (modified files are marked with a *)

## `src/`
Contains the source code of the project

#### `script_oversample.ipynb`*

## `src/results/`
Contains files and notebooks used to fine-tune the model and plot the results.

#### `bert_hyperparameter_tuning.py`
10-fold cross validation to find the best hyperparameters for the BERT model. Can be run to test hyperparameters individually or by running a grid search.

#### `bert_tuning_viz.ipynb`
Notebook to load the results of the hyperparameter tuning and plot them.

#### `results_plots.ipynb`
Notebook to load the results of the experiments and plot them.

## `src/configs/`
Contains the configuration file.

#### `config.yaml`*
Configuration parameters to be loaded in `script_oversample.py`.

## `src/test/`
Contains notebooks used for testing some implementations.

#### `test_BertHyperparametersTuning.ipynb`
Notebook to test the implementation of the crossvalidation for BERT hyperparameters tuning.

#### `test_BERTPipeline.ipynb`
Notebook to test the implementation of the BERTPipeline class.

#### `test_BERTs_Playground.ipynb`
Notebook to run and test the BERT model.

#### `test_Vectorisation_masking.ipynb`
Notebook to test the implementation of the Vectorisation and masking classes.

## `src/ml/BERT/`
Contains our implementation of the BERT model.

#### `BERT.py`
Functions that create the BERT model.

#### `BERTPipeline.py`
Functions to train and predict with the BERT model such that it can be implemented in the main pipeline.

#### `Config.py`
Configuration class for centralised management of simulation parameters.

#### `MaskedLanguageModel.py`
A masked language model that extends the `tf.keras.Model` class.

#### `MaskedTextGenerator.py`
Callback class for generating masked text predictions during training.

#### `Vectorisation.py`
Manages the vectorisation encoding and decoding of state-action sequences.
Every state-action sequence is transformed into a unique token, taking care of special tokens (padding, breaks). Encode-decode using either a dictionary or a numpy array.

#### `masking.py`
Generate masked input and corresponding labels for masked language modeling.

## `src/ml/samplers/`
Contains the different oversampling methods.

#### `synthetic_oversampler.py`*
This class oversamples the minority class to rebalance the distribution at 50/50. 
It takes all of the minority samples, and then randomly picks the other to fulfill the 50/50 criterion

# Figures
## `Fig.1` Pipeline used for the creation of synthetic data
![Pipeline used for the creation of synthetic data]<img src="https://github.com/CS-433/ml-project-2-losmodelosmatemagicos/blob/main/figures/BERT.png?raw=true" height="300">

## `Fig.2` Original distribution of the data and the 4 oversampling strategies.
![Distribution of the Data and the oversampling strategies](https://github.com/CS-433/ml-project-2-losmodelosmatemagicos/blob/main/figures/Oversampling_distribution.png?raw=true)

## `Fig.3` ROC score and FPR across experiments oversampling based on labels.
![ROC score and FPR across experiments oversampling based on labels](https://github.com/CS-433/ml-project-2-losmodelosmatemagicos/blob/main/figures/ROC%20scores%20and%20FPR%20Across%20Experiments.png?raw=true)

## `Fig.4` FPR with respect the language demographics across experiments.
![FPR with respect the language demographics across experiments](https://github.com/CS-433/ml-project-2-losmodelosmatemagicos/blob/main/figures/FPR_language.png?raw=true)
