import numpy as np
import pandas as pd
import logging
from typing import Tuple

from sklearn.model_selection import StratifiedKFold
from ml.splitters.splitter import Splitter
from collections import Counter

class MultipleStratifiedKSplit(Splitter):
    """Stratifier that splits the data into stratified fold

    Args:
        Splitter (Splitter): Inherits from the class Splitter
    """
    
    def __init__(self, settings:dict):
        super().__init__(settings)
        self._name = 'stratified k folds'
        self._notation = 'stratkf'
        self._random_seed = settings['seeds']['splitter']
        self._n_folds = self._settings['ml']['nfolds']
        
        self._settings = dict(settings)
        self.__init_splitter()
        
    def set_n_folds(self, n_folds):
        self._n_folds = n_folds
        
    def __init_splitter(self):
        print('    init splitter ', self._n_folds)
        print('    random seed splitter', self._random_seed)
        print(self._n_folds)
        self._splitter = StratifiedKFold(
            n_splits=self._n_folds,
            random_state=self._random_seed,
            shuffle=True
        )
        # logging.debug('splitter', self._splitter)
        
    def split(self, x:list, stratifier:list) -> Tuple[list, list]:
        return self._splitter.split(x, stratifier)

    def next_split(self, x, y):
        return next(self.split(x, y))
            
        
        