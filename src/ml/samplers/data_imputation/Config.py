from dataclasses import dataclass

@dataclass
class Config:
    NUM_STATES = 4
    NUM_ACTIONS = 6
    MAX_LEN = 256
    BATCH_SIZE = 32
    LR = 0.001
    VOCAB_SIZE = 30000
    EMBED_DIM = 128
    NUM_HEAD = 8  
    FF_DIM = 128 
    NUM_LAYERS = 1