import numpy as np
import pandas as pd
from collections import Counter
from typing import Tuple
from imblearn.over_sampling import RandomOverSampler as ros

class Sampler:
    """This class is used in the cross validation part, to change the distribution of the training data
    """
    
    def __init__(self, settings):
        self._name = 'sampler'
        self._notation = 'splr'
        self._settings = dict(settings)
        
    def get_name(self):
        return self._name

    def get_notation(self):
        return self._notation
    
    def sample(self, x: list, y: list) -> Tuple[list, list]:
        """This function changes the distribution of the data passed

        Args:
            x (list): features
            y (list): labels

        Returns:
            x_resampled (list): features with the new distribution
            y_resampled (list): labels for the rebalanced features
        """
        raise NotImplementedError

    def get_indices(self) -> list:
        """Returns the indexes chosen for the resampling

        Returns:
            list: indexes from the input
        """
        raise NotImplementedError

    def _equal_oversampling(self, sequences:list, oversampler:list, labels:list) -> Tuple[list, list]:
        """Oversamples based on some attributes determined in the config file (oversampler / oversampling_col)
        Rebalances all classes equally

        Args:
            sequences (list): behavioural sequences (features)
            oversampler (dict): column to use to oversample
            labels: labels linked to the machine learning part

        Returns:
            Tuple[list, list]: _description_
        """
        return self._oversample(sequences, labels, oversampler, 'all')

    def _major_oversampling(self, sequences:list, oversampler:list, labels:list) -> Tuple[list, list]:
        """Oversamples based on some attributes determined in the config file (oversampler / oversampling_col)
        Oversamples the majority class and kicks out all other instances from other classes

        Args:
            sequences (list): behavioural sequences (features)
            oversampler (dict): column to use to oversample
            labels: labels linked to the machine learning part

        Returns:
            Tuple[list, list]: _description_
        """
        distribution_os = Counter(oversampler)
        print('distribution os before the sampling: {}'.format(sorted(distribution_os.items())))
        
        sampler = {cluster: distribution_os[cluster] for cluster in distribution_os}

        majority_class = max(distribution_os, key=distribution_os.get)
        max_number = np.max([distribution_os[os_type] for os_type in distribution_os])
        sampler[majority_class] = max_number * self._settings['ml']['oversampler']['oversampling_factor']

        return self._oversample(sequences, labels, oversampler, sampler)   

    def _minor_oversampling(self, sequences:list, oversampler:list, labels:list) -> Tuple[list, list]:
        """Oversamples based on some attributes determined in the config file (oversampler / oversampling_col)
        Only oversamples the minority class

        Args:
            sequences (list): behavioural sequences (features)
            oversampler (dict): column to use to oversample
            labels: labels linked to the machine learning part

        Returns:
            Tuple[list, list]: _description_
        """
        distribution_os = Counter(oversampler)
        print('distribution os before the sampling: {}'.format(sorted(distribution_os.items())))
        
        sampler = {cluster: distribution_os[cluster] for cluster in distribution_os}

        minority_class = min(distribution_os, key=distribution_os.get)
        max_number = np.max([distribution_os[os_type] for os_type in distribution_os])
        sampler[minority_class] = max_number * self._settings['ml']['oversampler']['oversampling_factor']
        self._ros = ros(random_state=self._settings['seeds']['oversampler'], sampling_strategy=sampler)

        return self._oversample(sequences, labels, oversampler, sampler)

    def sample(self, sequences:list, oversampler:list, labels:list, demographics:list) -> Tuple[list, list]:
        """Chooses the mode of oversampling

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
        if self._settings['ml']['oversampler']['rebalancing_mode'] == 'equal_balancing':
            return self._equal_oversampling(sequences, oversampler, labels)

        elif 'major' in self._settings['ml']['oversampler']['rebalancing_mode']:
            return self._major_oversampling(sequences, oversampler, labels)
        
        elif 'minor' in self._settings['ml']['oversampler']['rebalancing_mode']:
            return self._minor_oversampling(sequences, oversampler, labels)


    def get_indices(self) -> np.array:
        return self._indices
