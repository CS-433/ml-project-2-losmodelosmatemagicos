import pickle
import numpy as np
import json

# This adds the BERT path to the python path, needed for the imports inside BERT modules
import sys
sys.path.append('./ml/BERT') 

from ml.BERT.Config import Config
from ml.BERT.Vectorisation import Vectorisation
import ml.BERT.masking as masking
import ml.BERT.BERT as BERT

import tensorflow as tf
from sklearn.model_selection import KFold
from itertools import product

def cross_validation(param=True):

    hyperparameters = {
        "EMBED_DIM": [32, 128, 256],
        "NUM_HEAD": [2, 4, 8],
        "FF_DIM": [32, 128, 256],
        "NUM_LAYERS": [1],
        "LR": [0.0001, 0.001, 0.01],
        "EPOCH": [10, 50, 100]
    }
    num_folds = 10

    with open("../data/ml4science_data.pkl", "rb") as fp:
        full_data = pickle.load(fp)

    data_list = [full_data["sequences"][i]["sequence"] for i in range(len(full_data["sequences"]))]

    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    best_hyperparameters = {}

    if param:
        for hyperparameter in hyperparameters.items():
            accuracy_seq_list = []
            accuracy_mask_list = []

            for value in hyperparameter[1]:
                print(f"\n\nTesting {hyperparameter[0]} with value {value}")

                match hyperparameter[0]:
                    case "EMBED_DIM":
                        config = Config(EMBED_DIM=value)
                    case "NUM_HEAD":
                        config = Config(NUM_HEAD=value)
                    case "FF_DIM":
                        config = Config(FF_DIM=value)
                    case "NUM_LAYERS":
                        config = Config(NUM_LAYERS=value)
                    case "LR":
                        config = Config(LR=value)
                    case "EPOCH":
                        config = Config(EPOCH=value)

                vectorisation = Vectorisation(config)
                accuracies_seq = []
                accuracies_mask = []

                for i, (train_index, test_index) in enumerate(kf.split(data_list)):
                    print("\n\n=========================================")
                    print(f"In fold {i + 1}\n")

                    train_data = [data_list[i] for i in train_index]
                    test_data = [data_list[i] for i in test_index]

                    seps_train = vectorisation.sep_from_seq(train_data)
                    seps_test = vectorisation.sep_from_seq(test_data)

                    train_data_encoded = vectorisation.encode(train_data, seps_train)
                    test_data_encoded = vectorisation.encode(test_data, seps_test)

                    x_masked_train, y_masked_train, sample_weights_train = masking.mask_input_and_labels(train_data_encoded, config.TOKEN_DICT, seed=32)
                    x_masked_test, y_masked_test, sample_weights_test = masking.mask_input_and_labels(test_data_encoded, config.TOKEN_DICT, seed=32)

                    mlm_ds_train = tf.data.Dataset.from_tensor_slices((x_masked_train, y_masked_train, sample_weights_train))
                    mlm_ds_train = mlm_ds_train.shuffle(1000).batch(config.BATCH_SIZE)
                    mlm_ds_test = tf.data.Dataset.from_tensor_slices((x_masked_test, y_masked_test, sample_weights_test))
                    mlm_ds_test = mlm_ds_test.shuffle(1000).batch(config.BATCH_SIZE)

                    bert_masked_model = BERT.create_masked_language_bert_model(config)
                    bert_masked_model.fit(mlm_ds_train, epochs=config.bert.epoch)

                    predictions = bert_masked_model.predict(x_masked_test)
                    predictions_max = np.argmax(predictions, axis=2)

                    accuracy_seq = np.sum((predictions_max == y_masked_test) * (y_masked_test != 0)) / np.sum(y_masked_test != 0)
                    accuracies_seq.append(accuracy_seq)
                    accuracy_mask = np.sum((predictions_max == y_masked_test) * (x_masked_test == config.TOKEN_DICT['[MASK]'])) / np.sum(x_masked_test == config.TOKEN_DICT['[MASK]'])
                    accuracies_mask.append(accuracy_mask)

                accuracy_seq_list.append(np.mean(accuracies_seq))
                accuracy_mask_list.append(np.mean(accuracies_mask))

            best_value_seq = hyperparameter[1][np.argmax(accuracy_seq_list)]
            best_accuracy_seq = np.max(accuracy_seq_list)
            best_value_mask = hyperparameter[1][np.argmax(accuracy_mask_list)]
            best_accuracy_mask = np.max(accuracy_mask_list)
            best_hyperparameters[hyperparameter[0]] = {"value_seq": best_value_seq, 
                                                       "accuracy_seq": best_accuracy_seq, 
                                                       "all_accuracies_seq": accuracy_seq_list, 
                                                       "value_mask": best_value_mask, 
                                                       "accuracy_mask": best_accuracy_mask, 
                                                       "all_accuracies_mask": accuracy_mask_list}

        print(f"The best hyperparameters and their values are:")
        for key, value in best_hyperparameters.items():
            print("For the sequence")
            print(f"{key}: {value['value_seq']} with an accuracy of {value['accuracy_seq']} and all accuracies: {value['all_accuracies_seq']}")
            print("For the mask")
            print(f"{key}: {value['value_mask']} with an accuracy of {value['accuracy_mask']}, and all accuracies: {value['all_accuracies_mask']}")

        # Save best hyperparameters to a file
        with open("ml/BERT/best_hyperparameters_no_combinations_best_param_default.json", "w") as file:
            json.dump(best_hyperparameters, file, indent=4)

    else:
        accuracy_seq_list = []
        accuracy_mask_list = []
        hyperparameters_combinations = []

        for hyperparameter in product(*hyperparameters.values()):
            print(f"\n\nTesting this combination: {hyperparameter}")
            config = Config(EMBED_DIM=hyperparameter[0], 
                            NUM_HEAD=hyperparameter[1], 
                            FF_DIM=hyperparameter[2], 
                            NUM_LAYERS=hyperparameter[3], 
                            LR=hyperparameter[4], 
                            EPOCH=hyperparameter[5]) # To change according to the hyperparameters to test
            vectorisation = Vectorisation(config)

            hyperparameters_combinations.append(hyperparameter)
            accuracies_seq = []
            accuracies_mask = []

            for i, (train_index, test_index) in enumerate(kf.split(data_list)):
                print("\n\n=========================================")
                print(f"In fold {i + 1}\n")
                train_data = [data_list[i] for i in train_index]
                test_data = [data_list[i] for i in test_index]

                seps_train = vectorisation.sep_from_seq(train_data)
                seps_test = vectorisation.sep_from_seq(test_data)

                train_data_encoded = vectorisation.encode(train_data, seps_train)
                test_data_encoded = vectorisation.encode(test_data, seps_test)

                x_masked_train, y_masked_train, sample_weights_train = masking.mask_input_and_labels(train_data_encoded, config.TOKEN_DICT, seed=32)
                x_masked_test, y_masked_test, sample_weights_test = masking.mask_input_and_labels(test_data_encoded, config.TOKEN_DICT, seed=32)

                mlm_ds_train = tf.data.Dataset.from_tensor_slices((x_masked_train, y_masked_train, sample_weights_train))
                mlm_ds_train = mlm_ds_train.shuffle(1000).batch(config.BATCH_SIZE)
                mlm_ds_test = tf.data.Dataset.from_tensor_slices((x_masked_test, y_masked_test, sample_weights_test))
                mlm_ds_test = mlm_ds_test.shuffle(1000).batch(config.BATCH_SIZE)

                bert_masked_model = BERT.create_masked_language_bert_model(config)
                bert_masked_model.fit(mlm_ds_train, epochs=config.bert.epoch)

                predictions = bert_masked_model.predict(x_masked_test)
                predictions_max = np.argmax(predictions, axis=2)

                accuracy_seq = np.sum((predictions_max == y_masked_test) * (y_masked_test != 0)) / np.sum(y_masked_test != 0)
                accuracies_seq.append(accuracy_seq)
                accuracy_mask = np.sum((predictions_max == y_masked_test) * (x_masked_test == config.TOKEN_DICT['[MASK]'])) / np.sum(x_masked_test == config.TOKEN_DICT['[MASK]'])
                accuracies_mask.append(accuracy_mask)

            accuracy_seq_list.append(np.mean(accuracies_seq))
            accuracy_mask_list.append(np.mean(accuracies_mask))

            with open("ml/BERT/hyperparameters_combinations", "a") as file:
                file.write("Hyperparameters combinations:\n")
                file.write(
                    f"EMBED_DIM: {hyperparameter[0]}\
                        \nNUM_HEAD: {hyperparameter[1]}\
                        \nFF_DIM: {hyperparameter[2]}\
                        \nNUM_LAYERS: {hyperparameter[3]}\
                        \nLR: {hyperparameter[4]}\
                        \nEPOCH: {hyperparameter[5]}\n"
                        )
                file.write(f"Accuracy for the sequence: {np.mean(accuracies_seq)}\n")
                file.write(f"Accuracy for the mask: {np.mean(accuracies_mask)}\n\n")

        print(f"The best hyperparameters values for the sequences are {hyperparameters_combinations[np.argmax(accuracy_seq_list)]} with an accuracy of {np.max(accuracy_seq_list)}")
        print(f"The best hyperparameters values for the masks are {hyperparameters_combinations[np.argmax(accuracy_mask_list)]} with an accuracy of {np.max(accuracy_mask_list)}")

        # Save best hyperparameters to a file
        with open("ml/BERT/best_hyperparameters_combinations", "a") as file:
            file.write(f"Best hyperparameters values for the sequences are {hyperparameters_combinations[np.argmax(accuracy_seq_list)]} with an accuracy of {np.max(accuracy_seq_list)}\n")
            file.write(f"Best hyperparameters values for the masks are {hyperparameters_combinations[np.argmax(accuracy_mask_list)]} with an accuracy of {np.max(accuracy_mask_list)}\n")


if __name__ == "__main__":
    cross_validation(param=False)