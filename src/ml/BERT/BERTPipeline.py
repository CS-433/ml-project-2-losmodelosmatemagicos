import numpy as np
import tensorflow as tf
import BERT
from Config import Config


class BERTPipeline:

    def __init__(self, config: Config):
        self.config = config

    def train(self, mlm_ds: tf.data.Dataset, verbose=0):
        """
        This function trains the model on the masked language model task.
        Args:
            sequences (list): list of sequences 
        """
        mlm_ds = mlm_ds.batch(self.config.BATCH_SIZE)

        bert_masked_model = BERT.create_masked_language_bert_model(self.config)
        #bert_masked_model.summary() # We make 10 CV so is too much to print the summary

        bert_masked_model.fit(mlm_ds, epochs=self.config.bert.epoch, verbose=verbose) # No need of callbacks 
        # verbose=0 to not print anything ; verbose=2 to see the progress bar 
        
        # If we want to save the model
        # bert_masked_model.save("bert_models/bert_mlm.keras") 

        self.model = bert_masked_model
    
    def predict(self, sequences: list, only_masked=True, verbose=0):
        """
        This function predicts (or at least try...) the masked tokens in the sequences, to create synthetic data.
        Args:
            sequences (array): masked and vectorised sequences
        Returns:
            predictions (list): list of predictions for the masked tokens
        """
        predictions = self.model.predict(sequences, verbose=verbose)
        predictions_max = np.argmax(predictions, axis=2)
        predictions_max = np.where(sequences==self.config.TOKEN_DICT['[PAD]'], self.config.TOKEN_DICT['[PAD]'], predictions_max)
        if only_masked: predictions_max = np.where(sequences==self.config.TOKEN_DICT['[MASK]'], predictions_max, sequences)

        return predictions_max