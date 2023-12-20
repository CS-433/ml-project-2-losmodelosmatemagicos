class Config:
    '''Configuration class for centralised management of simulation parameters'''
    
    def __init__(self, MAX_LEN=512, BATCH_SIZE=16, EMBED_DIM=32, NUM_HEAD=2, FF_DIM=32, NUM_LAYERS=1, LR=0.01, EPOCH=50,):
        self.MAX_LEN = MAX_LEN
        self.BATCH_SIZE = BATCH_SIZE
        self.TOKEN_DICT = {"[PAD]": 0, "[MASK]": 1, "[SEP]": 2}
        self.vectorisation = self.ConfigVectorisation()
        self.bert = self.ConfigBert(EMBED_DIM=EMBED_DIM, NUM_HEAD=NUM_HEAD, FF_DIM=FF_DIM, NUM_LAYERS=NUM_LAYERS, LR=LR, EPOCH=EPOCH)
        self.VOCAB_SIZE = self.vectorisation.NUM_ACTIONS * self.vectorisation.NUM_STATES + len(self.TOKEN_DICT)

    class ConfigVectorisation:
        def __init__(self, NUM_STATES=4, NUM_ACTIONS=6, SEP_IDX=8, SEP_LENGTH=15):
            self.NUM_STATES = NUM_STATES
            self.NUM_ACTIONS = NUM_ACTIONS
            self.SEP_IDX = SEP_IDX
            self.SEP_LENGTH = SEP_LENGTH

    class ConfigBert:
        def __init__(
            self, EMBED_DIM, NUM_HEAD, FF_DIM, NUM_LAYERS, LR, EPOCH
        ):
            self.EMBED_DIM = EMBED_DIM
            self.NUM_HEAD = NUM_HEAD
            self.FF_DIM = FF_DIM
            self.NUM_LAYERS = NUM_LAYERS
            self.LR = LR
            self.epoch = EPOCH
            self.train_per_mask = 0.15
            self.pred_per_mask = 0.15
            self.pred_only_masked = True
            self.RANDOM_BERT = 1 # for normal BERT, 1 for uniform random BERT, 2 for random BERT with sequence distribution density function

