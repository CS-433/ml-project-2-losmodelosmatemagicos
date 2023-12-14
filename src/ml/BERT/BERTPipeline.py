import numpy as np
import tensorflow as tf

import masking 
import BERT
from Config import Config
from Vectorisation import Vectorisation


class BertPipeline:

    def __init__(self, config: Config, vectorisation: Vectorisation):
        self.config = config
        self.vec = vectorisation

    def train(self, sequences: list):
        """
        This function trains the model on the masked language model task.
        Args:
            sequences (list): list of sequences 
        """
        # Sequences are already in list format
        sequences = self.vec.encode(sequences)

        x_masked_encoded, y_masked_encoded, sample_weights = masking.mask_input_and_labels(sequences, self.config.TOKEN_DICT)

        mlm_ds = tf.data.Dataset.from_tensor_slices((x_masked_encoded, y_masked_encoded, sample_weights))
        mlm_ds = mlm_ds.shuffle(1000).batch(self.config.BATCH_SIZE)

        bert_masked_model = BERT.create_masked_language_bert_model(self.config)
        bert_masked_model.summary()

        bert_masked_model.fit(mlm_ds, self.config.bert.epoch ) # No need of callbacks
        
        # No know yet if we want to save the model
        # bert_masked_model.save("bert_models/bert_mlm.keras") 

        return bert_masked_model
    
    def predict(self, sequences: list, bert_masked_model):
        """
        This function predicts (or at least try...) the masked tokens in the sequences, to create synthetic data.
        Args:
            sequences (list): list of sequences already 1-hot encoded
            bert_masked_model: the model to use for prediction
        Returns:
            predictions (list): list of predictions for the masked tokens
        """
        predictions = bert_masked_model.predict(sequences)
        predictions_max = np.argmax(predictions, axis=2)

        return predictions_max