import pickle
import numpy as np
import masking
import BERT
import tensorflow as tf
import sys
import json

sys.path.append("../../BERT/")
from Vectorisation import Vectorisation

from Config import Config
from sklearn.model_selection import KFold
from itertools import product
from MaskedTextGenerator import MaskedTextGenerator

def cross_validation(param=True):

    hyperparameters = {
        "EMBED_DIM": [32, 64, 128, 256],
        "NUM_HEAD": [2, 4, 8],
        "FF_DIM": [32, 64, 128, 256],
        "NUM_LAYERS": [1, 2, 4],
        "LR": [0.0001, 0.001, 0.01],
        "EPOCH": [10, 20, 30]
    }
    num_folds = 10

    with open("./ml4science_data.pkl", "rb") as fp:
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

                    train_data_encoded = vectorisation.encode(train_data)
                    test_data_encoded = vectorisation.encode(test_data)

                    x_masked_train, y_masked_train, sample_weights_train = masking.mask_input_and_labels(train_data_encoded, config.TOKEN_DICT, seed=32)
                    x_masked_test, y_masked_test, sample_weights_test = masking.mask_input_and_labels(test_data_encoded, config.TOKEN_DICT, seed=32)

                    mlm_ds_train = tf.data.Dataset.from_tensor_slices((x_masked_train, y_masked_train, sample_weights_train))
                    mlm_ds_train = mlm_ds_train.shuffle(1000).batch(config.BATCH_SIZE)
                    mlm_ds_test = tf.data.Dataset.from_tensor_slices((x_masked_test, y_masked_test, sample_weights_test))
                    mlm_ds_test = mlm_ds_test.shuffle(1000).batch(config.BATCH_SIZE)

                    bert_masked_model = BERT.create_masked_language_bert_model(config)
                    bert_masked_model.fit(mlm_ds_train, epochs=20, validation_data=mlm_ds_test)

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
        with open("best_hyperparameters_no_combinations.json", "w") as file:
            json.dump(best_hyperparameters, file, indent=4)

    else:
        accuracy_list = []
        hyperparameters_combinations = []

        for hyperparameter in product(*hyperparameters.values()):
            config = Config(NUM_LAYERS=hyperparameter[0], LR=hyperparameter[1]) # To change according to the hyperparameters to test
            vectorisation = Vectorisation(config)

            hyperparameters_combinations.append(hyperparameter)
            accuracies = []

            for train_index, test_index in kf.split(data_list):
                train_data = [data_list[i] for i in train_index]
                test_data = [data_list[i] for i in test_index]

                train_data_encoded = vectorisation.encode(train_data)
                test_data_encoded = vectorisation.encode(test_data)

                x_masked_train, y_masked_train, sample_weights_train = masking.mask_input_and_labels(train_data_encoded, config.TOKEN_DICT, seed=32)
                x_masked_test, y_masked_test, sample_weights_test = masking.mask_input_and_labels(test_data_encoded, config.TOKEN_DICT, seed=32)

                mlm_ds_train = tf.data.Dataset.from_tensor_slices((x_masked_train, y_masked_train, sample_weights_train))
                mlm_ds_train = mlm_ds_train.shuffle(1000).batch(config.BATCH_SIZE)
                mlm_ds_test = tf.data.Dataset.from_tensor_slices((x_masked_test, y_masked_test, sample_weights_test))
                mlm_ds_test = mlm_ds_test.shuffle(1000).batch(config.BATCH_SIZE)

                bert_masked_model = BERT.create_masked_language_bert_model(config)

                bert_masked_model.fit(mlm_ds_train, epochs=20, validation_data=mlm_ds_test)

                predictions = bert_masked_model.predict(x_masked_test)
                predictions_max = np.argmax(predictions, axis=2)

                accuracy = np.sum((predictions_max == y_masked_test) * (x_masked_test == config.TOKEN_DICT['[MASK]'])) / np.sum(x_masked_test == config.TOKEN_DICT['[MASK]'])
                accuracies.append(accuracy)

            accuracy_list.append(np.mean(accuracies))

        print(f"The best hyperparameters values for {list(hyperparameters.keys())} are {hyperparameters_combinations[np.argmax(accuracy_list)]} with an accuracy of {np.max(accuracy_list)}")
            

if __name__ == "__main__":
    cross_validation()