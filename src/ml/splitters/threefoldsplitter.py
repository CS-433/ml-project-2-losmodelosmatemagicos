import numpy as np
import pandas as pd
import logging
from typing import Tuple

from sklearn.model_selection import train_test_split
from ml.splitters.splitter import Splitter
from collections import Counter

class ThreeFoldSplitter(Splitter):
    """Stratifier that splits the data into stratified fold

    Args:
        Splitter (Splitter): Inherits from the class Splitter
    """
    
    def __init__(self, settings:dict):
        super().__init__(settings)
        self._name = '3 test-train folds'
        self._notation = '3ttf'
        self._random_seed = settings['seeds']['splitter']
        self._test_size = settings['ml']['test_size']
        self._n_folds = self._settings['ml']['nfolds']
        
        self._settings = dict(settings)
        self.__init_splitter()
        
    def set_n_folds(self, n_folds):
        self._n_folds = n_folds
        
    def __init_splitter(self):
        print('    init splitter ', self._n_folds)

    def split(self, x:list, stratifier:list,) -> Tuple[list, list]:
        train, test, _, _ = train_test_split(
            range(len(x)), stratifier, test_size=float(self._test_size), stratify=stratifier, random_state=np.random.randint(999)
        )
        return train, test

