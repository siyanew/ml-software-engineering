import os
from spacy.symbols import SPACE

# CONFIG

DEBUG = True
DATASET = 'demo'
DATASET_ENCODING = 'utf-8'

PREPROCESS_COMMIT_MSG_MIN_LEN = 10
PREPROCESS_COMMIT_MSG_MIN_WORDS = 2

PREPROCESS_SPACY_LANGUAGE_MODEL = 'en_core_web_sm'
PREPROCESS_IGNORE_POS = [SPACE]

# /CONFIG

# Directories
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
DB_DIR = os.path.join(ROOT_DIR, 'db')
