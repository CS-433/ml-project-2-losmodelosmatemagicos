import numpy as np
import tensorflow as tf
import BERT
from Config import Config


class BERTPipeline:

    def __init__(self, config: Config):
        self.config = config

    def train(self, mlm_ds: tf.data.Dataset):
        """
        This function trains the model on the masked language model task.
        Args:
            sequences (list): list of sequences 
        """
        mlm_ds = mlm_ds.batch(self.config.BATCH_SIZE)

        bert_masked_model = BERT.create_masked_language_bert_model(self.config)
        #bert_masked_model.summary() # We make 10 CV so is too much to print the summary

        bert_masked_model.fit(mlm_ds, epochs=self.config.bert.epoch, verbose=0) # No need of callbacks 
        # verbose=0 to not print anything ; verbose=2 to see the progress bar 
        
        # If we want to save the model
        # bert_masked_model.save("bert_models/bert_mlm.keras") 

        self.model = bert_masked_model
    
    def predict(self, sequences: list):
        """
        This function predicts (or at least try...) the masked tokens in the sequences, to create synthetic data.
        Args:
            sequences (list): list of sequences already 1-hot encoded
            bert_masked_model: the model to use for prediction
        Returns:
            predictions (list): list of predictions for the masked tokens
        """
        predictions = self.model.predict(sequences)
        predictions_max = np.argmax(predictions, axis=2)

        return predictions_max