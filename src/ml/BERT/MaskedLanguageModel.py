import tensorflow as tf
from tensorflow import keras

class MaskedLanguageModel(tf.keras.Model):
    """
    A masked language model that extends the `tf.keras.Model` class.
    
    Parameters:
        inputs (tf.Tensor): The input tensor for the model.
        outputs (tf.Tensor): The output tensor for the model.
        name (str): The name of the model.
        loss_fn (function): The loss function used for training.
        loss_tracker (tf.keras.metrics.Metric): The loss tracker metric.
    """
    
    def __init__(self, inputs, outputs, name):
        super(MaskedLanguageModel, self).__init__(inputs=inputs, outputs=outputs, name=name)
        self.loss_fn = keras.losses.SparseCategoricalCrossentropy(reduction=keras.losses.Reduction.NONE)
        self.loss_tracker = keras.metrics.Mean(name="loss")

    def train_step(self, inputs):
        """
        Performs a single training step for the masked language model.
        
        Parameters:
            inputs: The input data for the training step.
        
        Returns:
            dict: A dictionary mapping metric names to their current values.
        """
        if len(inputs) == 3:
            features, labels, sample_weight = inputs
        else:
            features, labels = inputs
            sample_weight = None

        with tf.GradientTape() as tape:
            predictions = self(features, training=True)
            loss = self.loss_fn(labels, predictions, sample_weight=sample_weight)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute our own metrics
        self.loss_tracker.update_state(loss, sample_weight=sample_weight)

        # Return a dict mapping metric names to current value
        return {"loss": self.loss_tracker.result()}

    @property
    def metrics(self):
        """
        Returns the list of metrics used by the model. 
        We list our `Metric` objects here so that `reset_states()` can be called automatically at the start of each epoch or at the start of `evaluate()`.
        If you don't implement this property, you have to call `reset_states()` yourself at the time of your choosing.
        
        Returns:
            list: A list of `tf.keras.metrics.Metric` objects.
        """
        return [self.loss_tracker]



