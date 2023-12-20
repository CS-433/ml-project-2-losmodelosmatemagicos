import numpy as np
import tensorflow as tf
import BERT
from Config import Config
import numpy as np


class BERTPipeline:
    """
    BERTPipeline class represents a pipeline for training and predicting with BERT models.
    """

    def __init__(self, config: Config):
        self.config = config

    def train(self, mlm_ds: tf.data.Dataset, verbose=0):
        """
        Trains the BERT model on the masked language model task.
        
        Args:
            mlm_ds (tf.data.Dataset): The dataset for masked language model task.
            verbose (int, optional): Verbosity mode. Defaults to 0.
        """

        if self.config.bert.RANDOM_BERT: return # If we want to use a random BERT, we don't need to train it

        mlm_ds = mlm_ds.batch(self.config.BATCH_SIZE)

        bert_masked_model = BERT.create_masked_language_bert_model(self.config)

        bert_masked_model.fit(mlm_ds, epochs=self.config.bert.epoch, verbose=verbose) # No need of callbacks 

        self.model = bert_masked_model
    
    def predict(self, sequences: np.array, only_masked=True, verbose=0):
        """
        Predicts the masked tokens in the sequences or the full sequence to create synthetic data.

        With "self.config.bert.RANDOM_BERT" defined by the instance Config we can also
        determines a random baselines to compare the performance of the BERT model.
        0 : normal BERT ; 1 : uniform random BERT ; 2 : random BERT with sequence distribution density function
        
        Args:
            sequences (np.array): Masked and vectorized sequences.
            only_masked (bool, optional): Whether to predict only the masked tokens or the full sequence, defaults to True.
            verbose (int, optional): Verbosity mode, defaults to 0.
        
        Returns:
            np.array: Predictions for the masked tokens or full sequence
        """

        # normal BERT
        if self.config.bert.RANDOM_BERT == 0: 
            predictions = self.model.predict(sequences, verbose=verbose)
            predictions_max = np.argmax(predictions, axis=2)
        
        # uniform random BERT
        elif self.config.bert.RANDOM_BERT == 1: 
            predictions_max = np.random.randint(3, self.config.VOCAB_SIZE, sequences.shape)

        # random BERT with sequence distribution density function
        elif self.config.bert.RANDOM_BERT == 2: 
            predictions_max = np.zeros_like(sequences)
            for i in range(sequences.shape[0]):
                unique, counts = np.unique(sequences[i], return_counts=True)
                
                special_token = np.isin(unique, np.array(list(self.config.TOKEN_DICT.values())))

                unique = np.delete(unique, special_token)
                counts = np.delete(counts, special_token)

                predictions_max[i] = np.random.choice(unique, size=sequences.shape[1], p=counts/np.sum(counts))

        # We don't want to predict the [PAD] tokens and want to (potentially) predict only the [MASK] tokens
        predictions_max = np.where(sequences==self.config.TOKEN_DICT['[PAD]'], self.config.TOKEN_DICT['[PAD]'], predictions_max)
        if only_masked: predictions_max = np.where(sequences==self.config.TOKEN_DICT['[MASK]'], predictions_max, sequences)

        return predictions_max