import numpy as np

class Preprocesing:
    """
    Class for preprocessing data: encoding and decoding.

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
    __init__(self, num_states: int, num_actions: int, token_dict: dict, sep: float = 0) -> None:
        Initializes the Preprocesing object with the given parameters.

    encode_vocabulary(self, data: dict):
        Encodes the vocabulary in the given data.

    encode_sep(self, data, index_break = 8):
        Encodes 

    decode_vocabulary(self, data):
        Decodes the vocabulary in the given data.
    """

    def __init__(self, num_states: int, num_actions: int, token_dict: dict, sep: float = 0) -> None:
        """
        Initializes the Preprocesing object with the given parameters.
        """

        self.ns = num_states
        self.na = num_actions
        self.token_dict = token_dict
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
        dict
            Encoded data.
        """

        encoded_data = data.copy()

        if self.sep > 0:
            self.encode_sep(encoded_data)

        for i in range(len(data['sequences'])):
            for j in range(len(data['sequences'][i]['sequence'])):
                if (data['sequences'][i]['sequence'][j] != self.token_dict['[SEP]']):
                    non_zero = np.nonzero(data['sequences'][i]['sequence'][j])
                    shift = self.ns - 1 - len(self.token_dict) # 4 - 1 - 3 = 0 in our case
                    value = non_zero[0][0] * self.na + non_zero[0][1] - shift # Actions index start at 1
                    encoded_data['sequences'][i]['sequence'][j] = value

        return encoded_data
    
    def encode_sep(self, data, index_break = 8):
        """
        Encodes the breaks if longer than this value "sep", in the given data.

        Parameters
        ----------
        data : dict
            Data to be encoded.
        index_break : int, optional
            Index break value, by default 8.
        """

        for i in range(len(data['sequences'])):
                for j in range(len(data['sequences'][i]['sequence'])):
                    if (((np.nonzero(data['sequences'][i]['sequence'][j]))[0][1] == [index_break]) and ((data['sequences'][0]['end'][j] - data['sequences'][0]['begin'][j]) > self.sep ) ): 
                        data['sequences'][i]['sequence'][j] = self.token_dict['[SEP]']

    def decode_vocabulary(self, data):
        """
        Decodes the vocabulary in the given data.

        Parameters
        ----------
        data : dict
            Data to be decoded.

        Returns
        -------
        dict
            Decoded data.
        """
        
        for i in range(len(data['sequences'])):
            for j in range(len(data['sequences'][i]['sequence'])):
                value = [0] * (self.ns + self.na)
                shift = self.ns - 1 - len(self.token_dict) # 4 - 1 - 3 = 0 in our case
                action_idx = (data['sequences'][i]['sequence'][j] - len(self.token_dict) - 1) % self.na + self.ns
                state_idx = (data['sequences'][i]['sequence'][j] - action_idx + shift) // self.na
                value[action_idx] = 1
                value[state_idx] = 1

                data['sequences'][i]['sequence'][j] = value

        return data