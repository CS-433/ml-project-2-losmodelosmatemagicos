from dataclasses import dataclass

@dataclass
class ConfigVectorisation:
    NUM_STATES: int = 4
    NUM_ACTIONS: int = 6
    TOKEN_DICT: dict = {'[PAD]': 0, '[MASK]': 1, '[SEP]': 2}
    SEP_IDX: int = 8
    SEP_LENGTH: int = 15  # Hyperparameter

@dataclass
class ConfigBert:
    EMBED_DIM: int = 128
    NUM_HEAD: int = 8
    FF_DIM: int = 128
    NUM_LAYERS: int = 1
    LR: float = 0.001

@dataclass
class Config:
    MAX_LEN: int = 256
    BATCH_SIZE: int = 32  
    VOCAB_SIZE: int = 30000 
    vectorisation: ConfigVectorisation = ConfigVectorisation()
    bert: ConfigBert = ConfigBert()