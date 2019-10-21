from typing import Iterable, List

from spacy.lang.char_classes import ALPHA
from spacy.lang.punctuation import TOKENIZER_INFIXES, TOKENIZER_PREFIXES, TOKENIZER_SUFFIXES
from spacy.language import Language
from spacy.symbols import POS, NORM, ORTH, X
from spacy.tokenizer import Tokenizer
from spacy.tokens import Token
from spacy.util import compile_prefix_regex, compile_suffix_regex, compile_infix_regex

from preprocessing.constants import PREPROCESS_DIFF_TOKEN_VERSION as TOK_VERSION

import re

# Diff tokenizer regex. Improved by considering pairs of punctuation (e.g. ++, //, +=) as one token
# Based on: https://www.nltk.org/api/nltk.tokenize.html (nltk.WordPunctTokenizer)
# re_punct_tokens = re.compile(r'\w+|[^\w\s]+')
re_punct_tokens = re.compile(r'/\*{1,2}|\*/|[-+*/><!]=|[+-/&|]{2}|\w+|[^\w\s]{1}')


def tokenize_diff(diff: str) -> List[str]:
    """
    Split diff string into tokens.
    """
    return re_punct_tokens.findall(diff)


def tokens_to_string(tokens: Iterable[Token]) -> str:
    """
    Glue tokens together, and expand contractions (e.g. "wouldn't" to "would not")
    """

    # N.B.: changing norm_ to something else here also influences utils.process.parse_commit_message()
    return ' '.join([token.norm_ for token in tokens])


def add_special_tokenizer_cases(nlp: Language) -> Language:
    infix_re = compile_infix_regex(tuple(TOKENIZER_INFIXES + [r"(?<=[{a}0-9])([()#\.]+|(-)+([->])+)(?=[{a}0-9])".format(a=ALPHA)]))
    prefix_re = compile_prefix_regex(tuple(TOKENIZER_PREFIXES + [r'^[.-]+']))
    suffix_re = compile_suffix_regex(tuple(TOKENIZER_SUFFIXES + [r'[.-]+$']))
    nlp.tokenizer = Tokenizer(nlp.vocab, prefix_search=prefix_re.search, suffix_search=suffix_re.search, infix_finditer=infix_re.finditer, token_match=None)

    for tok in ['==', '+=', '-=', '*=', '/=', '%=', '!=', '<>', '->', '-->', '--', '---', TOK_VERSION]:
        nlp.tokenizer.add_special_case(tok, [{ORTH: tok, NORM: tok, POS: X}])
    return nlp


def is_sha1(maybe_sha):
    """Source: https://stackoverflow.com/a/32234251"""

    if len(maybe_sha) != 40:
        return False
    try:
        sha_int = int(maybe_sha, 16)
    except ValueError:
        return False
    return True
