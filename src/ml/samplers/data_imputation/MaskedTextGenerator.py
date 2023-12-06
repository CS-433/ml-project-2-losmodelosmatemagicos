from tensorflow import keras
import numpy as np


class MaskedTextGenerator(keras.callbacks.Callback):
    """
    Callback class for generating masked text predictions during training.

    Args:
        sample_tokens (numpy.ndarray): The input tokens to generate predictions for.
        mask_token_id (int): The ID of the mask token in the token mapping dictionary.
        n_masks (int, optional): The number of mask predictions to print. Defaults to 5.
        top_k (int, optional): The number of top predictions to print for each mask. Defaults to 3.
    """

    def __init__(self, sample_tokens, mask_token_id, n_masks=5, top_k=3):
        self.n_masks = n_masks
        self.sample_tokens = sample_tokens
        self.mask_token_id = mask_token_id
        self.top_k = top_k

    def on_epoch_end(self, epoch, logs=None):
        """
        Callback method called at the end of each epoch.

        Generates masked text predictions and prints the results.

        Args:
            epoch (int): The current epoch number.
            logs (dict, optional): Dictionary of logs containing the training metrics. Defaults to None.
        """
        prediction = self.model.predict(self.sample_tokens)
        print('prediction shape:', prediction.shape)
        
        masked_index = np.where(self.sample_tokens[0] == self.mask_token_id) # find where the first student sequence was masked
        mask_prediction = prediction[0][masked_index] # get the predictions for the masked tokens shape=(#masked_inputs, VOCAB_SIZE)
        print('mask pred shape:', mask_prediction.shape)
        
        # getting the top_k predictions for the n_maks first masked predictions best_results.shape=(n_masks, top_k)
        best_results = mask_prediction[:self.n_masks].argsort()
        best_results = best_results[:, -self.top_k:][:, ::-1]
        print('best results shape:', best_results.shape)
        best_results_probs = mask_prediction[np.arange(self.n_masks).reshape(-1, 1), best_results] # getting the probabilities for the 5 first masked predictions

        '''print("\ninput_seq\n", self.sample_tokens[0])
        print("\npredictions\n", best_results)
        print("\nprobabilities\n", best_results_probs.round(2))'''

        # Print header
        print("\nmasked nb: \t ", end="")
        for i in range(self.n_masks):
            print(f"{i+1: <17}", end="")
        print()

        # Print predictions
        print("predictions: \t", end="")
        for i in range(self.n_masks):
            print(f"{best_results[i]}       ", end="")
        print()

        # Print probabilities
        print("probabilities: \t", end="")
        for i in range(self.n_masks):
            print(f"{best_results_probs[i].round(2)} ", end="")
        print('\n')

