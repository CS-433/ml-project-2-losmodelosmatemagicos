import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Tuple
from collections import Counter

from imblearn.over_sampling import RandomOverSampler as ros
from ml.samplers.sampler import Sampler

from ml.BERT.BERTPipeline import BERTPipeline
from ml.BERT.Config import Config
from ml.BERT.Vectorisation import Vectorisation
import ml.BERT.masking as masking

class SyntheticOversampler(Sampler):
    """This class oversamples the minority class to rebalance the distribution at 50/50. 
    It takes all of the minority samples, and then randomly picks the other to fulfill the 50/50 criterion

    Args:
        Sampler (Sampler): Inherits from the Sampler class
    """

    def __init__(self, settings):
        super().__init__(settings)
        self._name = "template oversampling"
        self._notation = "tempos"

        self._rebalancing_mode = self._settings["ml"]["oversampler"]["rebalancing_mode"]

    def _oversample(self, sequences: list, labels: list, oversampler: list, sampling_strategy: dict) -> Tuple[list, list, list]:
        """Oversamples x based on oversampler, according to the sampling_strategy. 
        There are 4 possible experiments; 
        o: sequence of demographic 1, O: SYNTHETIC sequence of demographic 1 // -: sequence of demographic 2, .: SYNTHETIC sequence of demographic 2
        1. [ooo] [-----] -> [oooooOOOOO] [-----.....] -> balanced demographics, 50% synthetic, 50% original -> os_half_half
        2. [ooo] [-----] -> [OOOOO] [.....]           -> balanced demographics, 100% synthetic              -> os_full_balanced
        3. [ooo] [-----] -> [OOO] [.....]             -> original demographics distribution, 100% synthetic -> os_full_og
        4. [ooo] [-----] -> [oooOO] [-----]           -> balanced demographics, rebalanced with synthetic   -> os_synth_rebalanced

        Args:
            sequences (list): sequences of interaction
            labels (list): target
            oversampler (list): list of the attributes by which to oversample, corresponding to the entries in x
            sampling_strategy (dict): dictionary with the keys as classes, and the values as number of samples to get, or str = 'all' if equally balanced

        Returns:
            shuffled_sequences: all the data you want to train with (right now, original data + synthetic data)
            shuffled labels: associated labels to the shuffled sequences
            shuffled indices: indices of the shuffled sequences (just to keep track what data comes from where. ( Optional, for reproducibility )
        """
        experiment = self._settings['experiment']['type']
        print(f'Running experiment {experiment}')

        print("distribution demographics before the sampling: {}".format(sorted(Counter(oversampler).items())))
        print("distribution labels before the sampling: {}".format(sorted(Counter(labels).items())))
        assert len(labels) == len(sequences)

        if experiment==1: ss, sl, si, so = self._os_half_half(sequences, labels, oversampler, sampling_strategy)
        if experiment==2: ss, sl, si, so = self._os_full_balanced(sequences, labels, oversampler, sampling_strategy)
        if experiment==3: ss, sl, si, so = self._os_full_og(sequences, labels, oversampler, sampling_strategy)
        if experiment==4: ss, sl, si, so = self._os_synth_rebalance(sequences, labels, oversampler, sampling_strategy)

        print("distribution demographics after the sampling: {}".format(sorted(Counter(so).items())))
        print("distribution labels after sampling: {}".format(sorted(Counter(sl).items())))

        return ss, sl, si

        
    def _os_half_half(self, sequences: list, labels: list, oversampler: list, sampling_strategy: dict) -> Tuple[list, list, list]:
        '''1. [ooo] [-----] -> [oooooOOOOO] [-----.....] -> balanced demographics, 50% synthetic, 50% original -> os_half_half'''

        self._ros = ros(
            random_state=self._settings["seeds"]["oversampler"],
            sampling_strategy=sampling_strategy,
        )

        indices = [[idx] for idx in range(len(sequences))]
        indices_resampled, _ = self._ros.fit_resample(indices, oversampler) # rebalanced data [ooo] [-----] -> [ooooo] [-----]

        # potential_shuffles contains the full resampled data
        potential_shuffles = [idx[0] for idx in indices_resampled]

        # 2) Objects storing the sequences which you will edit. Here, I called it shuffled because I shuffled the sequences, but you will do other nicer things than shuffles (hopefully ;))
        shuffled_sequences = []
        shuffled_oversampler = []
        shuffled_labels = []
        shuffled_indices = []

        config = Config()
        vec = Vectorisation(config)

        train_sequences = sequences
        train_seps = vec.sep_from_seq(train_sequences)
        encoded_sequences = vec.encode(train_sequences, train_seps)

        x_tr, y_tr, w_tr = masking.mask_input_and_labels(encoded_sequences, config.TOKEN_DICT, ratio_mask_per_seq = config.bert.train_per_mask)
        mlm_ds = tf.data.Dataset.from_tensor_slices((x_tr, y_tr, w_tr))

        bert = BERTPipeline(config)
        bert.train(mlm_ds)

        for idx in potential_shuffles: 
            # Vectorisation and masking of predicted sequences
            pred_sequences = [sequences[idx]]
            pred_seps = vec.sep_from_seq(pred_sequences)
            pred_encoded_sequences = vec.encode(pred_sequences, pred_seps)

            x_pred, *_ = masking.mask_input_and_labels(pred_encoded_sequences, config.TOKEN_DICT, ratio_mask_per_seq = config.bert.pred_per_mask)

            # Predicting the new sequences
            pred = bert.predict(x_pred, only_masked=config.bert.pred_only_masked)
            decoded_pred = vec.decode(pred)
            decoded_pred = vec.add_time_info(decoded_pred, pred_sequences)

            # Adding synthetic sequences
            shuffled_sequences.extend(decoded_pred)
            shuffled_labels.append(labels[idx])
            shuffled_indices.append(idx)
            shuffled_oversampler.append(oversampler[idx])

            # Adding the original BALANCED sequences
            shuffled_sequences.append(sequences[idx])
            shuffled_labels.append(labels[idx])
            shuffled_indices.append(idx)
            shuffled_oversampler.append(oversampler[idx])

        return shuffled_sequences, shuffled_labels, shuffled_indices, shuffled_oversampler
    
    def _os_full_balanced(self, sequences: list, labels: list, oversampler: list, sampling_strategy: dict) -> Tuple[list, list, list]:
        ''' 2. [ooo] [-----] -> [OOOOO] [.....] -> balanced demographics, 100% synthetic -> os_full_balanced'''

        self._ros = ros(
            random_state=self._settings["seeds"]["oversampler"],
            sampling_strategy=sampling_strategy,
        )

        indices = [[idx] for idx in range(len(sequences))]
        indices_resampled, _ = self._ros.fit_resample(indices, oversampler)

        # potential_shuffles contains the full resampled data
        potential_shuffles = [idx[0] for idx in indices_resampled]

        # 2) Objects storing the sequences which you will edit. Here, I called it shuffled because I shuffled the sequences, but you will do other nicer things than shuffles (hopefully ;))
        shuffled_sequences = []
        shuffled_oversampler = []
        shuffled_labels = []
        shuffled_indices = []

        config = Config()
        vec = Vectorisation(config)

        train_sequences = sequences
        train_seps = vec.sep_from_seq(train_sequences)
        encoded_sequences = vec.encode(train_sequences, train_seps)

        x_tr, y_tr, w_tr = masking.mask_input_and_labels(encoded_sequences, config.TOKEN_DICT, ratio_mask_per_seq = config.bert.train_per_mask)
        mlm_ds = tf.data.Dataset.from_tensor_slices((x_tr, y_tr, w_tr))

        bert = BERTPipeline(config)
        bert.train(mlm_ds)

        for idx in potential_shuffles: 
            # Vectorisation and masking of predicted sequences
            pred_sequences = [sequences[idx]]
            pred_seps = vec.sep_from_seq(pred_sequences)
            pred_encoded_sequences = vec.encode(pred_sequences, pred_seps)

            x_pred, *_ = masking.mask_input_and_labels(pred_encoded_sequences, config.TOKEN_DICT, ratio_mask_per_seq = config.bert.pred_per_mask)

            # Predicting the new sequences
            pred = bert.predict(x_pred, only_masked=config.bert.pred_only_masked)
            decoded_pred = vec.decode(pred)
            decoded_pred = vec.add_time_info(decoded_pred, pred_sequences)

            # Adding synthetic sequences
            shuffled_sequences.extend(decoded_pred)
            shuffled_labels.append(labels[idx])
            shuffled_indices.append(idx)
            shuffled_oversampler.append(oversampler[idx])

        return shuffled_sequences, shuffled_labels, shuffled_indices, shuffled_oversampler
    
    def _os_full_og(self, sequences: list, labels: list, oversampler: list, sampling_strategy: dict) -> Tuple[list, list, list]:
        ''' 3. [ooo] [-----] -> [OOO] [.....] -> original demographics distribution, 100% synthetic -> os_full_og'''

        # potential_shuffles contains the original unbalanced data
        indices = [[idx] for idx in range(len(sequences))]
        potential_shuffles = [idx[0] for idx in indices]

        # 2) Objects storing the sequences which you will edit. Here, I called it shuffled because I shuffled the sequences, but you will do other nicer things than shuffles (hopefully ;))
        shuffled_sequences = []
        shuffled_oversampler = []
        shuffled_labels = []
        shuffled_indices = []

        config = Config()
        vec = Vectorisation(config)

        train_sequences = sequences
        train_seps = vec.sep_from_seq(train_sequences)
        encoded_sequences = vec.encode(train_sequences, train_seps)

        x_tr, y_tr, w_tr = masking.mask_input_and_labels(encoded_sequences, config.TOKEN_DICT, ratio_mask_per_seq = config.bert.train_per_mask)
        mlm_ds = tf.data.Dataset.from_tensor_slices((x_tr, y_tr, w_tr))

        bert = BERTPipeline(config)
        bert.train(mlm_ds)

        for idx in potential_shuffles: 
            # Vectorisation and masking of predicted sequences
            pred_sequences = [sequences[idx]]
            pred_seps = vec.sep_from_seq(pred_sequences)
            pred_encoded_sequences = vec.encode(pred_sequences, pred_seps)

            x_pred, *_ = masking.mask_input_and_labels(pred_encoded_sequences, config.TOKEN_DICT, ratio_mask_per_seq = config.bert.pred_per_mask)

            # Predicting the new sequences
            pred = bert.predict(x_pred, only_masked=config.bert.pred_only_masked)
            decoded_pred = vec.decode(pred)
            decoded_pred = vec.add_time_info(decoded_pred, pred_sequences)

            # Adding synthetic sequences
            shuffled_sequences.extend(decoded_pred)
            shuffled_labels.append(labels[idx])
            shuffled_indices.append(idx)
            shuffled_oversampler.append(oversampler[idx])

        return shuffled_sequences, shuffled_labels, shuffled_indices, shuffled_oversampler
    
    def _os_synth_rebalance(self, sequences: list, labels: list, oversampler: list, sampling_strategy: dict) -> Tuple[list, list, list]:
        '''4. [ooo] [-----] -> [oooOO] [-----] -> balanced demographics, rebalanced with synthetic   -> os_synth_rebalanced'''

        self._ros = ros(
            random_state=self._settings["seeds"]["oversampler"],
            sampling_strategy=sampling_strategy,
        )

        indices = [[idx] for idx in range(len(sequences))]
        indices_resampled, _ = self._ros.fit_resample(indices, oversampler) # rebalanced data [ooo] [-----] -> [ooooo] [-----]

        # potential_shuffles contains only the oversampled part but not the initial data -> only rebalanced is synthetic
        potential_shuffles = [idx[0] for idx in indices_resampled]
        [potential_shuffles.remove(idx) for idx in range(len(sequences))]
        assert len(potential_shuffles) == (len(indices_resampled) - len(indices))

        # 2) Objects storing the sequences which you will edit. Here, I called it shuffled because I shuffled the sequences, but you will do other nicer things than shuffles (hopefully ;))
        shuffled_sequences = []
        shuffled_oversampler = []
        shuffled_labels = []
        shuffled_indices = []

        config = Config()
        vec = Vectorisation(config)

        train_sequences = sequences
        train_seps = vec.sep_from_seq(train_sequences)
        encoded_sequences = vec.encode(train_sequences, train_seps)

        x_tr, y_tr, w_tr = masking.mask_input_and_labels(encoded_sequences, config.TOKEN_DICT, ratio_mask_per_seq = config.bert.train_per_mask)
        mlm_ds = tf.data.Dataset.from_tensor_slices((x_tr, y_tr, w_tr))

        bert = BERTPipeline(config)
        bert.train(mlm_ds)

        for idx in potential_shuffles: 
            # Vectorisation and masking of predicted sequences
            pred_sequences = [sequences[idx]]
            pred_seps = vec.sep_from_seq(pred_sequences)
            pred_encoded_sequences = vec.encode(pred_sequences, pred_seps)

            x_pred, *_ = masking.mask_input_and_labels(pred_encoded_sequences, config.TOKEN_DICT, ratio_mask_per_seq = config.bert.train_per_mask)

            # Predicting the new sequences
            pred = bert.predict(x_pred, only_masked=config.bert.pred_only_masked)
            decoded_pred = vec.decode(pred)
            decoded_pred = vec.add_time_info(decoded_pred, pred_sequences)

            # Adding synthetic sequences
            shuffled_sequences.extend(decoded_pred)
            shuffled_labels.append(labels[idx])
            shuffled_indices.append(idx)
            shuffled_oversampler.append(oversampler[idx])

        # Adding the original (not balanced) sequences
        [shuffled_sequences.append(sequences[idx]) for idx in range(len(sequences))]
        [shuffled_labels.append(labels[idx]) for idx in range(len(labels))]
        [shuffled_indices.append(idx) for idx in range(len(labels))]
        [shuffled_oversampler.append(oversampler[idx]) for idx in range(len(oversampler))]

        return shuffled_sequences, shuffled_labels, shuffled_indices, shuffled_oversampler


    def sample(self, sequences: list, oversampler: list, labels: list, demographics: list) -> Tuple[list, list]:
        """Chooses the mode of oversampling
        Functions are right now in the sampler.py file

        1. equal oversampling: All instances are oversampled by n, determined by imbalanced-learn
        2. Major oversampling: Only the largest class is oversampled
        3. Only Major Oversampling: Only the largest class is oversampled, all other classes are taken out the training set
        4. Minor oversampling: Only the smallest class is oversampled
        5. Only Minor Oversampling: Only the smallest class is oversampled, all other classes are taken out the training set

        Args:
            sequences (list): behavioural sequences (features)
            oversampler (dict): column to use to oversample
            labels: labels linked to the machine learning part

        Returns:
            Tuple[list, list]: _description_
        """
        if self._settings["ml"]["oversampler"]["rebalancing_mode"] == "equal_balancing":
            return self._equal_oversampling(sequences, oversampler, labels)

        elif "major" in self._settings["ml"]["oversampler"]["rebalancing_mode"]:
            return self._major_oversampling(sequences, oversampler, labels)

        elif "minor" in self._settings["ml"]["oversampler"]["rebalancing_mode"]:
            return self._minor_oversampling(sequences, oversampler, labels)

    def get_indices(self) -> np.array:
        return self._indices
