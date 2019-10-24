import os
from spacy.symbols import SPACE

# RUN CONFIGURATION
DEBUG = False
DATASET = 'demo'
OUTPUT_ENCODING = 'utf-8'
SPACY_LANGUAGE_MODEL = 'en_core_web_sm'

# DIRECTORIES
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'data')

# ---

# MSG - FILTER
PREPROCESS_COMMIT_MSG_MIN_LEN = 5

PREPROCESS_COMMIT_MSG_MIN_TOKENS = 2
PREPROCESS_COMMIT_MSG_MAX_TOKENS = 30

# MSG - TOKEN TO STRING
PREPROCESS_IGNORE_POS = [SPACE]

# DIFFS - FILTER
PREPROCESS_DIFF_MIN_BYTES = 12
PREPROCESS_DIFF_MAX_BYTES = 1024 ** 2

PREPROCESS_DIFF_MAX_TOKENS = 100

# DIFFS - PROCESSING
PREPROCESS_DIFF_KEEP_EXTENSIONS = ['java', 'cs', 'md', 'txt']
PREPROCESS_DIFF_TOKEN_ADD = 'aaaaa'
PREPROCESS_DIFF_TOKEN_DEL = 'ddddd'
PREPROCESS_DIFF_TOKEN_VERSION = 'vvvvv'
PREPROCESS_DIFF_TOKEN_FILE = 'fffff'
