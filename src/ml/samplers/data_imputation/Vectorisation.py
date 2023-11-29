import numpy as np
from Config import Config

class Vectorisation:
    """
    Manages the vectorisation encoding and decoding of state-action sequences

    Attributes
    ----------
    num_states : int
        Number of states in the data.
    num_actions : int
        Number of actions in the data.
    token_dict : dict
        Dictionary mapping tokens to their corresponding indices.
    sep : float, optional
        Used to represent breaks if longer than this value , by default 0 (no breaks).

    Methods
    -------
    encode_vocabulary(self, data: dict):
        Encodes the vocabulary in the given data.

    encode_sep(self, data, index_break = 8):
        Encodes 

    decode_vocabulary(self, data):
        Decodes the vocabulary in the given data.
    """

    def __init__(self, config: Config, special_token_dict: dict, sep: float = 0) -> None:
        """
        Initializes the Preprocesing object with the given parameters.
        """

        self.config = config
        self.token_dict = special_token_dict
        self.sep = sep

    def encode_vocabulary(self, data: dict):
        """
        Encodes the vocabulary in the given data.

        Parameters
        ----------
        data : dict
            Data to be encoded.

        Returns
        -------
        np.array
            Encoded data.
        """

        # make a copy of the data to be able to return sampled data in the same shape
        self.data = data.copy()

        # if possible, shorten the sequences to avoid unnecessary padding
        student_seq = [len(data['sequences'][i]['sequence']) for i in range(len(data['sequences']))]
        self.config.MAX_LEN = min(self.config.MAX_LEN, max(student_seq))

        # this automatically adds zero padding at the end of the sequence
        encoded_data = np.zeros(shape=(len(data['sequences']), self.config.MAX_LEN))

        if self.sep > 0:
            self.encode_sep(encoded_data)

        for i in range(len(data['sequences'])):
            for j in range(len(data['sequences'][i]['sequence'])):
                if (data['sequences'][i]['sequence'][j] != self.token_dict['[SEP]']):
                    non_zero = np.nonzero(data['sequences'][i]['sequence'][j])
                    shift = self.ns - 1 - len(self.token_dict) # 4 - 1 - 3 = 0 in our case
                    value = non_zero[0][0] * self.na + non_zero[0][1] - shift # Actions index start at 1
                    encoded_data[i][j] = value

        return encoded_data
    
    def encode_sep(self, data: dict, break_idx: int = 8):
        """
        Encodes the breaks if longer than this value "sep", in the given data.

        Parameters
        ----------
        data : dict
            Data to be encoded.
        index_break : int, optional
            Index break value, by default 8.
        """

        students = data['sequences']
        for i in range(len(students)):
                sequences = students[i]['sequence']
                for j in range(len(sequences)):
                    is_break_idx = np.nonzero(sequences[j])[0][1] == break_idx
                    is_long_break = students[i]['end'][j] - students[i]['begin'][j] > self.sep
                    if (is_break_idx and is_long_break): 
                        sequences[j] = self.token_dict['[SEP]']

    def decode_vocabulary(self, synth_data: np.array):
        """
        Decodes the vocabulary in the given data.

        Parameters
        ----------
        synth_data : np.array
            Data to be decoded.

        Returns
        -------
        dict
            Decoded data.
        """
        
        for i in range(len(self.data['sequences'])):
            for j in range(len(self.data['sequences'][i]['sequence'])):
                value = [0] * (self.ns + self.na)
                shift = self.ns - 1 - len(self.token_dict) # 4 - 1 - 3 = 0 in our case
                action_idx = (synth_data[i][j] - len(self.token_dict) - 1) % self.na + self.ns
                state_idx = (synth_data[i][j] - action_idx + shift) // self.na
                value[action_idx] = 1
                value[state_idx] = 1

                self.data['sequences'][i]['sequence'][j] = value

        return self.data