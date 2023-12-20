"""
This file uses modified code from the "End-to-end Masked Language Modeling with BERT" 
originally authored by Ankur Singh, available at https://github.com/keras-team/keras-io/blob/master/examples/nlp/masked_language_modeling.py
and is licensed under the Apache License, Version 2.0.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from Config import Config
from MaskedLanguageModel import MaskedLanguageModel
import numpy as np


def bert_module(config: Config, query, key, value, i):
    """
    BERT module that performs multi-headed self-attention and feed-forward layer operations.

    Args:
        config (Config): Configuration object.
        query: Query tensor.
        key: Key tensor.
        value: Value tensor.
        i (int): Index of the module.

    Returns:
        sequence_output: Output tensor after applying self-attention and feed-forward layer operations.
    """
    # Multi headed self-attention
    attention_output = layers.MultiHeadAttention(
        num_heads=config.bert.NUM_HEAD,
        key_dim=config.bert.EMBED_DIM // config.bert.NUM_HEAD,
        name="encoder_{}/multiheadattention".format(i))(query, key, value)
    attention_output = layers.Dropout(0.1, name="encoder_{}/att_dropout".format(i))(attention_output)
    attention_output = layers.LayerNormalization(epsilon=1e-6, name="encoder_{}/att_layernormalization".format(i))(query + attention_output)

    # Feed-forward layer
    ffn = keras.Sequential(
        [
            layers.Dense(config.bert.FF_DIM, activation="relu"),
            layers.Dense(config.bert.EMBED_DIM),
        ],
        name="encoder_{}/ffn".format(i))
    
    ffn_output = ffn(attention_output)
    ffn_output = layers.Dropout(0.1, name="encoder_{}/ffn_dropout".format(i))(ffn_output)
    sequence_output = layers.LayerNormalization(epsilon=1e-6, name="encoder_{}/ffn_layernormalization".format(i))(attention_output + ffn_output)

    return sequence_output


def get_pos_encoding_matrix(max_len, d_emb):
    """
    Generate the positional encoding matrix for the BERT model.

    Args:
        max_len (int): Maximum length of the input sequence.
        d_emb (int): Dimension of the embedding.

    Returns:
        numpy.ndarray: The positional encoding matrix of shape (max_len, d_emb).
    """
    pos_enc = np.array(
        [
            [pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)]
            if pos != 0
            else np.zeros(d_emb)
            for pos in range(max_len)
        ]
    )
    pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])  # dim 2i
    pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])  # dim 2i+1
    return pos_enc

def create_masked_language_bert_model(config: Config):
    inputs = layers.Input((config.MAX_LEN,), dtype=tf.int64)

    word_embeddings = layers.Embedding(config.VOCAB_SIZE, config.bert.EMBED_DIM, name="word_embedding")(inputs)

    position_embeddings = layers.Embedding(
        input_dim=config.MAX_LEN,
        output_dim=config.bert.EMBED_DIM,
        weights=[get_pos_encoding_matrix(config.MAX_LEN, config.bert.EMBED_DIM)],
        name="position_embedding",
    )(tf.range(start=0, limit=config.MAX_LEN, delta=1))

    embeddings = word_embeddings + position_embeddings

    encoder_output = embeddings
    for i in range(config.bert.NUM_LAYERS):
        encoder_output = bert_module(config, encoder_output, encoder_output, encoder_output, i)

    mlm_output = layers.Dense(config.VOCAB_SIZE, name="mlm_cls", activation="softmax")(encoder_output)
    mlm_model = MaskedLanguageModel(
        inputs,
        mlm_output,
        name="masked_bert_model"
    )

    optimizer = keras.optimizers.Adam(learning_rate=config.bert.LR)
    mlm_model.compile(optimizer=optimizer)

    return mlm_model