# Dataset config
TRAIN_SRC_PATH = ""
TRAIN_TGT_PATH = ""
DEV_SRC_PATH = ""
DEV_TGT_PATH = ""
TEST_SRC_PATH = ""
TEST_TGT_PATH = ""

# Tokenizer config
VOCAB_SIZE = 16000
SPM_MODEL_PATH = "spm.model"
PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3

# Training config
SEED = None
BATCH_SIZE = 32
NUM_EPOCHS = 10
LAST_CHECKPOINT_PATH = "last_checkpoint.pt"
BEST_MODEL_PATH = "best_model.pt"
LEARNING_RATE = 3e-4
DROPOUT_RATE = 0.25
RESUME_TRAINING = False

# Model Config
MODEL_TYPE = "seq2seq"
MODEL_DIM = 256
HEAD_DIM = 64
SELECTIVE = False
FORGET = False
NUM_LAYERS = 4

# Infer config
PRUNE_TUNING_STEP = 4
MAX_NEW_TOKENS = 250
SELECTIVE_BUDGET = [None] * 4
PREDS_PATH = "preds.csv"