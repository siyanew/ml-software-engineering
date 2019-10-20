from typing import Iterable, List

from spacy.language import Language
from spacy.symbols import POS, SYM, NORM, ORTH
from spacy.tokens import Token

import re

# Source: https://www.nltk.org/api/nltk.tokenize.html (nltk.WordPunctTokenizer)
# re_punct_tokens = re.compile(r'\w+|[^\w\s]+')

# Better splitter (source: https://stackoverflow.com/a/43094210)
re_punct_tokens = re.compile(r"\w+(?:'\w+)?|[^\w\s]")


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
    nlp.tokenizer.add_special_case('==', [{ORTH: '==', NORM: '==', POS: SYM}])
    nlp.tokenizer.add_special_case('+=', [{ORTH: '+=', NORM: '+=', POS: SYM}])
    nlp.tokenizer.add_special_case('-=', [{ORTH: '-=', NORM: '-=', POS: SYM}])
    nlp.tokenizer.add_special_case('*=', [{ORTH: '*=', NORM: '*=', POS: SYM}])
    nlp.tokenizer.add_special_case('/=', [{ORTH: '/=', NORM: '/=', POS: SYM}])
    nlp.tokenizer.add_special_case('%=', [{ORTH: '%=', NORM: '%=', POS: SYM}])
    nlp.tokenizer.add_special_case('!=', [{ORTH: '!=', NORM: '!=', POS: SYM}])
    nlp.tokenizer.add_special_case('<>', [{ORTH: '<>', NORM: '<>', POS: SYM}])
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
