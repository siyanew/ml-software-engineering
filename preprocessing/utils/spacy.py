from typing import Iterable, List

from spacy.language import Language
from spacy.symbols import POS, SYM, NORM, ORTH
from spacy.tokens import Token

import re

from preprocessing.constants import PREPROCESS_DIFF_TOKEN_ADD as PREP_TOK_ADD, \
    PREPROCESS_DIFF_TOKEN_DEL as PREP_TOK_DEL, PREPROCESS_DIFF_TOKEN_VERSION as PREP_TOK_VER

# Source: https://www.nltk.org/api/nltk.tokenize.html (nltk.WordPunctTokenizer)
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

    # N.B.: changing norm_ to something else here also influences utils.process.parse_commit_message()
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
    nlp.tokenizer.add_special_case(PREP_TOK_ADD, [{ORTH: PREP_TOK_ADD, NORM: PREP_TOK_ADD, POS: SYM}])
    nlp.tokenizer.add_special_case(PREP_TOK_DEL, [{ORTH: PREP_TOK_DEL, NORM: PREP_TOK_DEL, POS: SYM}])
    nlp.tokenizer.add_special_case(PREP_TOK_VER, [{ORTH: PREP_TOK_VER, NORM: PREP_TOK_VER, POS: SYM}])


def is_sha1(maybe_sha):
    """Source: https://stackoverflow.com/a/32234251"""

    if len(maybe_sha) != 40:
        return False
    try:
        sha_int = int(maybe_sha, 16)
    except ValueError:
        return False
    return True
