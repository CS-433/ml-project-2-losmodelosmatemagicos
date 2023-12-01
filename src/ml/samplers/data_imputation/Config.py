class Config:
    '''Configuration class for centralised management of simulation parameters'''
    
    def __init__(self, MAX_LEN=256, BATCH_SIZE=32, VOCAB_SIZE=30000):
        self.MAX_LEN = MAX_LEN
        self.BATCH_SIZE = BATCH_SIZE
        self.VOCAB_SIZE = VOCAB_SIZE
        self.TOKEN_DICT = {"[PAD]": 0, "[MASK]": 1, "[SEP]": 2}
        self.vectorisation = self.ConfigVectorisation()
        self.bert = self.ConfigBert()

    class ConfigVectorisation:
        def __init__(self, NUM_STATES=4, NUM_ACTIONS=6, SEP_IDX=8, SEP_LENGTH=15):
            self.NUM_STATES = NUM_STATES
            self.NUM_ACTIONS = NUM_ACTIONS
            self.SEP_IDX = SEP_IDX
            self.SEP_LENGTH = SEP_LENGTH

    class ConfigBert:
        def __init__(
            self, EMBED_DIM=128, NUM_HEAD=8, FF_DIM=128, NUM_LAYERS=1, LR=0.001
        ):
            self.EMBED_DIM = EMBED_DIM
            self.NUM_HEAD = NUM_HEAD
            self.FF_DIM = FF_DIM
            self.NUM_LAYERS = NUM_LAYERS
            self.LR = LR
