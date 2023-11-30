import numpy as np

def get_masked_input_and_labels(encoded_texts: np.array ,special_token_dict: dict, ratio_mask_per_seq: float = 0.15, ratio_seq_masked: float = 0.9, ratio_random_seq: float = 0.1):
    """
    Generate masked input and corresponding labels for masked language modeling.

    Parameters
    ----------
    encoded_texts : np.array
        Array of encoded texts where each element represents a sequence of tokens.
    special_token_dict : dict
        Dictionary mapping special tokens to their corresponding token IDs.
    ratio_mask_per_seq : float, optional
        The ratio of WordPiece tokens to be masked in each sequence, by default 0.15.
    ratio_seq_masked : float, optional
        The ratio of masked tokens to be replaced with the [MASK] token, by default 0.9.
    ratio_random_seq : float, optional
        The ratio of masked tokens to be replaced with a random token, by default 0.1.

    Returns
    -------
    encoded_texts_masked : np.array
        Array of masked input sequences.
    y_labels : np.array
        Array of labels corresponding to the masked input sequences.
    sample_weights : np.array
        Array of sample weights to be used during training.

    """
    # we mask 15% of all WordPiece tokens in each sequence at random (values by default)
    inp_mask = np.random.rand(*encoded_texts.shape) < ratio_mask_per_seq
    # Do not mask special tokens
    inp_mask[encoded_texts <= len(special_token_dict) - 1 ] = False # The special_token_dict starts at 0

    # Set targets to -1 by default, it means ignore
    labels = -1 * np.ones(encoded_texts.shape, dtype=int)
    # Set labels for masked tokens
    labels[inp_mask] = encoded_texts[inp_mask]

    # Prepare input
    encoded_texts_masked = np.copy(encoded_texts)
    # Set input to [MASK] for the 90% of tokens ,this means leaving 10% unchanged (values by default)
    inp_mask_2mask = inp_mask & (np.random.rand(*encoded_texts.shape) < ratio_seq_masked)
    encoded_texts_masked[
        inp_mask_2mask
    ] = special_token_dict['[MASK]']

    # Set 10% to a random token of the 90% tokens modified, aka 10%/90% = 1/9 (values by default)
    inp_mask_2random = inp_mask_2mask & (np.random.rand(*encoded_texts.shape) < (ratio_random_seq/ratio_seq_masked) )
    encoded_texts_masked[inp_mask_2random] = np.random.randint(len(special_token_dict), max(encoded_texts)+1 , inp_mask_2random.sum() )# low = 3 ( included ) /// high = last_token + 1 (not included)

    # Prepare sample_weights to pass to .fit() method
    sample_weights = np.ones(labels.shape)
    sample_weights[labels == -1] = 0

    # y_labels would be same as encoded_texts i.e input tokens
    y_labels = np.copy(encoded_texts)

    return encoded_texts_masked, y_labels, sample_weights