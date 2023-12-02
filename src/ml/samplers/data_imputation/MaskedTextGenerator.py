from tensorflow import keras
import numpy as np
from pprint import pprint
import Vectorisation


class MaskedTextGenerator(keras.callbacks.Callback):
    """
    Callback class for generating masked text predictions during training.

    Args:
        sample_tokens (numpy.ndarray): The input tokens to generate predictions for.
        mask_token_id (int): The ID of the mask token in the token mapping dictionary.
        mapping_dict (dict): A dictionary mapping token IDs to their corresponding tokens.
        top_k (int, optional): The number of top predictions to consider. Defaults to 5.
    """

    def __init__(self, sample_tokens, mask_token_id, top_k=5):
        self.sample_tokens = sample_tokens
        self.mask_token_id = mask_token_id
        self.k = top_k

    def on_epoch_end(self, epoch, logs=None):
        """
        Callback method called at the end of each epoch.

        Generates masked text predictions and prints the results.

        Args:
            epoch (int): The current epoch number.
            logs (dict, optional): Dictionary of logs containing the training metrics. Defaults to None.
        """
        prediction = self.model.predict(self.sample_tokens)

        masked_index = np.where(self.sample_tokens == self.mask_token_id)
        masked_index = masked_index[1]
        mask_prediction = prediction[0][masked_index]

        top_indices = mask_prediction[0].argsort()[-self.k :][::-1]
        values = mask_prediction[0][top_indices]

        for i in range(len(top_indices)):
            p = top_indices[i]
            v = values[i]
            tokens = np.copy(self.sample_tokens[0])
            tokens[masked_index[0]] = p
            result = {
                "input_text": Vectorisation.decode_dict(self.sample_tokens[0]),
                "prediction": Vectorisation.decode_dict(tokens),
                "probability": v,
                "predicted mask token id": p,
            }
            pprint(result)

