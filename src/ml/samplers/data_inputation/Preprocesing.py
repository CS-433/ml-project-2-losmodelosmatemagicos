import numpy as np

class Preprocesing:

    def __init__(self, num_states, num_actions, token_dict, sep = 0):
        self.num_states = num_states
        self.num_actions = num_actions
        self.token_dict = token_dict
        self.sep = sep
    
    def encode_sep(self, data, sep, index_break = 8):

        for i in range(len(data['sequences'])):
                for j in range(len(data['sequences'][i]['sequence'])):
                    if ( ( (np.nonzero(data['sequences'][i]['sequence'][j]))[0][1] == [8]) and ( (data['sequences'][0]['end'][j] - data['sequences'][0]['begin'][j]) > self.sep ) ): 
                        data['sequences'][i]['sequence'][j] = self.token_dict['[SEP]']


    def encode_vocabulary(self, data):

        if self.sep > 0:
            self.encode_sep(data,self.sep)

        for i in range(len(data['sequences'])):
            for j in range(len(data['sequences'][i]['sequence'])):
                if (data['sequences'][i]['sequence'][j] != self.token_dict['[SEP]']):
                    idx = np.nonzero(data['sequences'][i]['sequence'][j])
                    value = idx[0][0]*self.num_actions + idx[0][1] - (self.num_states - len(self.token_dict) -1) #Want actions index start at 1
                    data['sequences'][i]['sequence'][j] = value

        return data
    
    def decode_vocabulary(self, data):
        raise NotImplementedError
        for i in range(len(data['sequences'])):
            for j in range(len(data['sequences'][i]['sequence'])):
                # Yannick do you modulo stuff here