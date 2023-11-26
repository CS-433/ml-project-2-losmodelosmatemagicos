import numpy as np

class Preprocesing:

    def __init__(self, num_states: int, num_actions: int, token_dict: dict, sep: float = 0) -> None:
        self.ns = num_states
        self.na = num_actions
        self.token_dict = token_dict
        self.sep = sep

    def encode_vocabulary(self, data: dict):

        encoded_data = data.copy()

        if self.sep > 0:
            self.encode_sep(encoded_data)

        for i in range(len(data['sequences'])):
            for j in range(len(data['sequences'][i]['sequence'])):
                if (data['sequences'][i]['sequence'][j] != self.token_dict['[SEP]']):
                    non_zero = np.nonzero(data['sequences'][i]['sequence'][j])
                    value = non_zero[0][0] * self.na + non_zero[0][1] - (self.ns - len(self.token_dict) - 1) # Actions index start at 1
                    encoded_data['sequences'][i]['sequence'][j] = value

        return encoded_data
    
    def encode_sep(self, data, index_break = 8):

        for i in range(len(data['sequences'])):
                for j in range(len(data['sequences'][i]['sequence'])):
                    if (((np.nonzero(data['sequences'][i]['sequence'][j]))[0][1] == [index_break]) and ((data['sequences'][0]['end'][j] - data['sequences'][0]['begin'][j]) > self.sep ) ): 
                        data['sequences'][i]['sequence'][j] = self.token_dict['[SEP]']

    def decode_vocabulary(self, data):
        
        for i in range(len(data['sequences'])):
            for j in range(len(data['sequences'][i]['sequence'])):
                value = [0] * (self.ns + self.na)
                shift = self.ns - len(self.token_dict) - 1 # 4 - 3 - 1 = 0 in our case
                action_idx = (data['sequences'][i]['sequence'][j] + shift) % self.na
                state_idx = (data['sequences'][i]['sequence'][j] + shift) // self.na
                value[action_idx] = 1
                value[state_idx] = 1

                data['sequences'][i]['sequence'][j] = value

        return data