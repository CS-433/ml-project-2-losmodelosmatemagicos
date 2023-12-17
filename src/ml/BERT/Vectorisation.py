import numpy as np
import copy
from Config import Config


class Vectorisation:
    """
    Manages the vectorisation encoding and decoding of state-action sequences.
    Every state-action sequence is transformed into a unique token, taking care of special tokens (padding, breaks).
 
    Workflow:        [ dict -> ] list -> np.array -> list [ -> dict ]
    Dict structure:  dict['sequences'][i]['sequence'][j]) where (i, j) = (num_students, num_sequences)
                                         ['begin'][j]) and ['end'][j]) being others keys of dict['sequences'][i]
    List structure:  nested list of vectors shape = (num_students, num_sequences, ns + na)
    Array structure: np.array of shape = (num_students, MAX_LEN)

    Attributes
    ----------
    config : The configuration object containing vectorisation parameters.
    ns : The number of states
    na : The number of actions

    Methods
    -------
    encode_dict:    dict -> np.array encoding
    decode_dict:    np.array -> dict decoding
    seps_from_dict: Returns sequence location of special token [SEP] (from dict)
    sep_from_seq:   Returns sequence location of special token [SEP] (from list)
    encode:         list -> np.array encoding
    decode:         np.array -> list decoding
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.ns = self.config.vectorisation.NUM_STATES
        self.na = self.config.vectorisation.NUM_ACTIONS

    def encode_dict(self, decoded_data: dict) -> np.array:
        """ dict -> np.array encoding. Special token SEP is automatically encoded. A dict is saved for decoding. """

        # make a copy of the data to be able to return sampled data in the same shape
        self.data_dict = copy.deepcopy(decoded_data)

        if self.config.vectorisation.SEP_LENGTH > 0:
            seps = self.seps_from_dict(self.data_dict)
        else:
            seps = None

        # extracting the sequences from the data as a list of lists (#students, #sequences, #states + #actions)
        data_as_list = [
            self.data_dict["sequences"][i]["sequence"]
            for i in range(len(self.data_dict["sequences"]))
        ]
        encoded_data = self.encode(data_as_list, seps)

        return encoded_data

    def decode_dict(self, encoded_data: np.array) -> dict:
        """ np.array -> dict decoding """

        decoded_data = self.decode(encoded_data)
        for i in range(len(self.data_dict["sequences"])):
            self.data_dict["sequences"][i]["sequence"] = decoded_data[i]

        return self.data_dict

    def seps_from_dict(self, decoded_data: dict) -> np.array:
        """
        Encodes the breaks longer than the specified config.SEP_LENGTH from a dict.

        Returns: Boolean array shape = (num_students, MAX_LEN). True if sequence is a break and False otherwise.
        """
        seps = np.zeros(
            shape=(len(decoded_data["sequences"]), self.config.MAX_LEN), dtype=bool
        )

        for i in range(len(decoded_data["sequences"])):
            stud = decoded_data["sequences"][i]
            sequences = stud["sequence"]
            for j in range(
                min(len(sequences), self.config.MAX_LEN)
            ):  # avoids overshoot if max sequence length > MAX_LEN
                is_break_idx = (
                    np.nonzero(sequences[j])[0][1] == self.config.vectorisation.SEP_IDX
                )
                is_long_break = (
                    stud["end"][j] - stud["begin"][j]
                    > self.config.vectorisation.SEP_LENGTH
                )
                if is_break_idx and is_long_break:
                    seps[i][j] = True
        return seps
    
    def sep_from_seq(self, decoded_data):
        """
        Encodes the breaks longer than the specified config.SEP_LENGTH from a list of sequences.

        Returns: Boolean array shape = (num_students, MAX_LEN). True if sequence is a break and False otherwise.
        """
        seps = np.zeros(shape=(len(decoded_data), self.config.MAX_LEN), dtype=bool)

        for i in range(len(decoded_data)):
            sequences = decoded_data[i]
            for j in range(min(len(sequences), self.config.MAX_LEN)):  # avoids overshoot if max sequence length > MAX_LEN
                action_idx = np.nonzero(sequences[j])[0][1]
                is_break_idx = (action_idx == self.config.vectorisation.SEP_IDX)
                is_long_break = sequences[j][action_idx] > self.config.vectorisation.SEP_LENGTH

                if is_break_idx and is_long_break:
                    seps[i][j] = True
        return seps

    def encode(self, decoded_data: list, seps: np.array = None) -> np.array:
        """ list -> np.array encoding. If boolean array for breaks is provided, SEP tokens are encoded"""

        # this automatically adds zero padding at the end of the sequence
        encoded_data = np.zeros(
            shape=(len(decoded_data), self.config.MAX_LEN), dtype=int
        )
        assert seps is None or encoded_data.shape == seps.shape

        # assigns a unique token to every combination of state and action
        for stud_idx, stud in enumerate(decoded_data):
            non_zero = np.nonzero(stud)
            shift = self.ns - len(self.config.TOKEN_DICT)  # 4 - 3 = 1 in our case
            value = non_zero[-1][0::2] * self.na + non_zero[-1][1::2] - shift
            encoded_data[stud_idx, : len(value)] = value[
                : self.config.MAX_LEN
            ]  # no risk of overshoot if max sequence length > MAX_LEN

        # encoding seps after the rest to avoid overwriting
        if seps is not None:
            encoded_data[seps] = self.config.TOKEN_DICT["[SEP]"]

        return encoded_data

    def decode(self, encoded_data: np.array) -> list:
        """ np.array -> list decoding """
        
        num_stud, num_seq = encoded_data.shape
        assert num_seq == self.config.MAX_LEN

        decoded_data = []

        for i in range(num_stud):
            stud_decoded_data = []
            for j in range(num_seq):
                value = [0] * (self.ns + self.na)
                if encoded_data[i][j] == self.config.TOKEN_DICT["[PAD]"]:
                    break  # end of sequence if padding is reached
                if (
                    encoded_data[i][j] == self.config.TOKEN_DICT["[SEP]"]
                ):  # handles special token for breaks
                    value[self.config.vectorisation.SEP_IDX] = 1
                    # no previous action -> state is 3; previous action -> state unchanged after break
                    if len(stud_decoded_data) == 0:
                        value[self.ns - 1] = 1
                    else:
                        value[: self.ns] = stud_decoded_data[-1][: self.ns]
                else:
                    shift = self.ns - len(
                        self.config.TOKEN_DICT
                    )  # 4 - 3 = 1 in our case
                    action_idx = (
                        encoded_data[i][j] - len(self.config.TOKEN_DICT)
                    ) % self.na + self.ns
                    state_idx = (encoded_data[i][j] - action_idx + shift) // self.na
                    value[action_idx] = 1
                    value[state_idx] = 1

                stud_decoded_data.append(value)
            decoded_data.append(stud_decoded_data)

        return decoded_data

    def add_time_info(self, decoded_data: list, original_seq: list) -> list:
        """ list -> list. Adds time information to the sequences. """
