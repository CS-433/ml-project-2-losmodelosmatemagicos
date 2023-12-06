from tensorflow import keras
import numpy as np


class MaskedTextGenerator(keras.callbacks.Callback):
    """
    Callback class for generating masked text predictions during training.

    Args:
        sample_tokens (numpy.ndarray): The input tokens to generate predictions for.
        mask_token_id (int): The ID of the mask token in the token mapping dictionary.
        top_k (int, optional): The number of top predictions to consider. Defaults to 5.
    """

    def __init__(self, sample_tokens, mask_token_id, top_k=10):
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
        print(prediction.shape)

        '''masked_index = np.where(self.sample_tokens[0] == self.mask_token_id)
        mask_prediction = prediction[0][masked_index]
        print(mask_prediction.shape)

        
        top_indices = mask_prediction[0].argsort()[-self.k :][::-1]
        print(top_indices)
        values = mask_prediction[0][top_indices]

        p = top_indices[:self.k]
        v = values[:self.k]
        result = {
            "input_seq": self.sample_tokens[0],
            "prediction": p,
            "probability": v
        }
        pprint(result)'''

        
        masked_index = np.where(self.sample_tokens[0] == self.mask_token_id) # find where the first student sequence was masked
        mask_prediction = prediction[0][masked_index] # get the predictions for the masked tokens shape=(#masked_inputs, VOCAB_SIZE)
        print('mask pred shape:', mask_prediction.shape)
        
        best_results = mask_prediction[:self.k].argmax(axis=1) # getting max prediction for the 5 first masked predictions
        best_results_probs = mask_prediction[np.arange(self.k), best_results] # getting the probabilities for the 5 first masked predictions

        print("input_seq\n", self.sample_tokens[0])
        print("predictions\n", best_results)
        print("max probability\n", [round(prob, 2) for prob in best_results_probs])

