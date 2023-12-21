import pickle
from pathlib import PurePath
import numpy as np

class DataLoader:

    def __init__(self, settings):
        self._settings = dict(settings)

    def _select_data(self):
        """Reads the full data files. Data files are dictionaries where each index contains a new dictionary with:
        - the behavioural sequence (already as a feature)
        - the demographics information
        - the label
        Additionally, the full data also contains an entry 'available_demographics', with all of the keys accessing the demographics 

        Returns:
           full data
        """
        ################################ CHEMLAB ################################
        ##### Features #####
        path = '../data/ml4science_data_fake.pkl'
        path = PurePath(path)
        with open(path, 'rb') as fp:
            full_data = pickle.load(fp)
        return full_data

    def _format_data(self, idd:dict):
        """Takes the dictionary with the big data (as formatted and explained in self._select_data), and returns the data into machine learning friendly lists

        Args:
            full_data (dict): as described in _select_data

        Returns:
            sequences (list): list of all of the behavioural sequences of the students, already formatted,
            labels (list): list of the labels for the ml part
            demographics (dict): dictionary where each of the keys is a demographic information, and the values are the list. 

        
        """
        full_data = dict(idd['sequences'])
        indices = list(full_data.keys())
        demographics_list = idd['available_demographics']
        label = 'label'

        sequences = []
        labels = []
        demographics = {key:[] for key in demographics_list}
        stratifiers = []
        oversamplers = []
        oversample_map = {}

        for idx in indices:
            # x, y
            sequences.append(full_data[idx][self._settings['data']['key']])
            labels.append(full_data[idx][label])

            # demographics
            for demo in demographics_list:
                d = str(full_data[idx][demo])
                if demo in self._settings['data']['others']:
                    for o in self._settings['data']['others'][demo]:
                        d = d.replace(str(o), 'other')
                demographics[demo].append(d)

            # splitter
            strat = ''
            for s in self._settings['ml']['splitter']['stratifier_col']:
                if s == 'label':
                    strat += str(full_data[idx][label])
                elif s in demographics:
                    strat += demographics[s][-1]
                else:
                    strat += str(full_data[idx][s])
            stratifiers.append(strat)

            # os
            oversampler = ''
            for os in self._settings['ml']['oversampler']['oversampling_col']:
                if os == 'label':
                    oversampler += str(full_data[idx][label])
                elif os in demographics:
                    oversampler += demographics[os][-1]
                else:
                    oversampler += str(full_data[idx][os])
            oversamplers.append(oversampler)
            if oversampler not in oversample_map:
                oversample_map[oversampler] = []
            oversample_map[oversampler].append(full_data[idx][label])

        demographics['stratifier_col'] = [s for s in stratifiers]
        demographics['oversampler_col'] = [s for s in oversamplers]

        return sequences, labels, demographics


    def load_data(self):
        full_data = self._select_data()
        sequences, labels, demographics = self._format_data(full_data)
        return sequences, labels, demographics