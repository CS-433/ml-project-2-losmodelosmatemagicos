import logging
from typing import Tuple

from ml.models.model import Model
from ml.models.ts_attention import TimestepAttentionModel

from ml.samplers.sampler import Sampler
from ml.samplers.random_oversampler import RandomOversampler
from ml.samplers.no_sampler import NoSampler
from ml.samplers.template_synthetic_oversampler import TemplateOversampler

from ml.scorers.scorer import Scorer
from ml.scorers.binaryclassification_scorer import BinaryClfScorer

from ml.splitters.splitter import Splitter
from ml.splitters.threefoldsplitter import ThreeFoldSplitter
from ml.splitters.stratified_split import MultipleStratifiedKSplit

from ml.crossvalidators.crossvalidator import CrossValidator
from ml.crossvalidators.threefoldvalidator import ThreeCrossValidator
from ml.crossvalidators.nonnested_cv import NonNestedRankingCrossVal

class MLPipeline:
    """This script assembles the machine learning component and creates the training pipeline according to:
    
        - splitter
        - sampler
        - model
        - xvalidator
        - scorer
    """
    
    def __init__(self, settings:dict):
        logging.debug('initialising the xval')
        self._name = 'training maker'
        self._notation = 'trnmkr'
        self._settings = dict(settings)
        self._experiment_root = self._settings['experiment']['root_name']
        self._experiment_name = settings['experiment']['name']
        
        self._build_pipeline()
    

    def get_sampler(self):
        return self._sampler

    def get_scorer(self):
        return self._scorer

    def get_model(self):
        return self._model

    def _choose_splitter(self) -> Splitter:
        self._splitter = MultipleStratifiedKSplit
        return self._splitter
    
    def _choose_sampler(self):
        if self._settings['ml']['oversampler']['mode'] == 'ros':
            self._sampler = RandomOversampler
        if self._settings['ml']['oversampler']['mode'] == 'augmentation':
            self._sampler = TemplateOversampler
        if self._settings['ml']['oversampler']['mode'] == 'none':
            self._sampler = NoSampler
            
    def _choose_model(self):
        self._model = TimestepAttentionModel

    def _get_num_classes(self):
        self._n_classes = 2
        self._settings['experiment']['n_classes'] = self._n_classes

    def _choose_scorer(self):
        self._scorer = BinaryClfScorer

    def _choose_xvalidator(self):
        self._gridsearch = {}
        self._xval = NonNestedRankingCrossVal(self._settings, self._splitter, self._sampler, self._model, self._scorer)
                
    def _build_pipeline(self):
        self._choose_splitter()
        self._choose_sampler()
        self._choose_model()
        self._choose_scorer()
        self._choose_xvalidator()
        
    def train(self, sequences:list, labels:list, demographics:list):
        results = self._xval.xval(sequences, labels, demographics)
        