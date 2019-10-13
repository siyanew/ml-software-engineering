from typing import Iterable, List

from spacy.language import Language
from spacy.symbols import POS, SYM, NORM, ORTH
from spacy.tokens import Token

import re

# Source: https://www.nltk.org/api/nltk.tokenize.html (nltk.WordPunctTokenizer)
from preprocessing.constants import PREPROCESS_DIFF_TOKEN_ADD, PREPROCESS_DIFF_TOKEN_DEL

re_punct_tokens = re.compile(r'\w+|[^\w\s]+')


def tokenize_diff(diff: str) -> List[str]:
    """
    Split diff string into tokens.
    """

    # TODO: might be a better way to tokenize diffs than the NLTK WordPunctTokenizer approach
    return re_punct_tokens.findall(diff)


def tokens_to_string(tokens: Iterable[Token]) -> str:
    """
    Glue tokens together, and expand contractions (e.g. "wouldn't" to "would not")
    """

    return ' '.join([token.norm_ for token in tokens])


def _add_special_tokenizer_cases(nlp: Language) -> None:
    nlp.tokenizer.add_special_case('==', [{ORTH: '==', NORM: '==', POS: SYM}])
    nlp.tokenizer.add_special_case('+=', [{ORTH: '+=', NORM: '+=', POS: SYM}])
    nlp.tokenizer.add_special_case('-=', [{ORTH: '-=', NORM: '-=', POS: SYM}])
    nlp.tokenizer.add_special_case('*=', [{ORTH: '*=', NORM: '*=', POS: SYM}])
    nlp.tokenizer.add_special_case('/=', [{ORTH: '/=', NORM: '/=', POS: SYM}])
    nlp.tokenizer.add_special_case('%=', [{ORTH: '%=', NORM: '%=', POS: SYM}])
    nlp.tokenizer.add_special_case('!=', [{ORTH: '!=', NORM: '!=', POS: SYM}])
    nlp.tokenizer.add_special_case('<>', [{ORTH: '<>', NORM: '<>', POS: SYM}])
    nlp.tokenizer.add_special_case(PREPROCESS_DIFF_TOKEN_ADD,
                                   [{ORTH: PREPROCESS_DIFF_TOKEN_ADD, NORM: PREPROCESS_DIFF_TOKEN_ADD, POS: SYM}])
    nlp.tokenizer.add_special_case(PREPROCESS_DIFF_TOKEN_DEL,
                                   [{ORTH: PREPROCESS_DIFF_TOKEN_DEL, NORM: PREPROCESS_DIFF_TOKEN_DEL, POS: SYM}])


def is_sha1(maybe_sha):
    """Source: https://stackoverflow.com/a/32234251"""

    if len(maybe_sha) != 40:
        return False
    try:
        sha_int = int(maybe_sha, 16)
    except ValueError:
        return False
    return True
