import numpy as np


def mask_input_and_labels(
    encoded_data: np.array,
    special_token_dict: dict,
    ratio_mask_per_seq: float = 0.15,
    ratio_seq_masked: float = 0.9,
    ratio_random_seq: float = 0.1,
):
    """
    Generate masked input and corresponding labels for masked language modeling.

    Parameters
    ----------
    encoded_data :
        Array of encoded texts where each element represents a sequence of tokens.
    special_token_dict : 
        Dictionary mapping special tokens to their corresponding token IDs.
    ratio_mask_per_seq :
        The ratio of WordPiece tokens to be masked in each sequence, by default 0.15.
    ratio_seq_masked :
        The ratio of masked tokens to be replaced with the [MASK] token, by default 0.9.
    ratio_random_seq :
        The ratio of masked tokens to be replaced with a random token, by default 0.1.

    Returns
    -------
    encoded_data_masked : np.array
        Array of masked input sequences.
    y_labels : np.array
        Array of labels corresponding to the masked input sequences.
    sample_weights : np.array
        Array of sample weights to be used during training.

    """
    # we mask 15% of all sequence tokens for each student at random (values by default)
    mask_idx = np.random.rand(*encoded_data.shape) < ratio_mask_per_seq

    # Do not mask special tokens (special_token_dict starts at 0)
    mask_idx[encoded_data < len(special_token_dict)] = False 

    # Set targets to -1 by default, it means ignore
    labels = -1 * np.ones(encoded_data.shape, dtype=int)
    # Set labels for masked tokens
    labels[mask_idx] = encoded_data[mask_idx]

    # Set input to [MASK] for the 90% of tokens, this means leaving 10% unchanged (values by default)
    encoded_data_masked = np.copy(encoded_data)
    mask_idx_2mask = mask_idx & (np.random.rand(*encoded_data.shape) < ratio_seq_masked)
    encoded_data_masked[mask_idx_2mask] = special_token_dict["[MASK]"]

    # Set 10% to a random token of the total tokens but form the 90% tokens modified, aka 10%/90% = 1/9 (values by default)
    mask_idx_2random = mask_idx_2mask & (np.random.rand(*encoded_data.shape) < (ratio_random_seq / ratio_seq_masked))
    encoded_data_masked[mask_idx_2random] = np.random.randint(
        len(special_token_dict),  # low = len(special_token_dict) ( included, is not a problem cuz special_token_dict starts at 0 )
        np.max(encoded_data) + 1,  # high = last_token + 1 (not included)
        mask_idx_2random.sum(),
    )

    # Prepare sample_weights to pass to .fit() method
    sample_weights = np.ones(labels.shape)
    sample_weights[labels == -1] = 0

    # y_labels would be same as encoded seqs i.e input tokens
    y_labels = np.copy(encoded_data)

    return encoded_data_masked, y_labels, sample_weights
