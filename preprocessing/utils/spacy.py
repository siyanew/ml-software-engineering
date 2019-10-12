from typing import Iterable, List

from spacy.attrs import ORTH
from spacy.language import Language
from spacy.symbols import POS, SYM
from spacy.tokens import Token

import re

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

    return ' '.join([token.norm_ for token in tokens])


def _add_special_tokenizer_cases(nlp: Language) -> None:
    nlp.tokenizer.add_special_case('==', [{ORTH: '==', POS: SYM}])
    nlp.tokenizer.add_special_case('+=', [{ORTH: '==', POS: SYM}])
    nlp.tokenizer.add_special_case('-=', [{ORTH: '==', POS: SYM}])
    nlp.tokenizer.add_special_case('*=', [{ORTH: '==', POS: SYM}])
    nlp.tokenizer.add_special_case('/=', [{ORTH: '==', POS: SYM}])
    nlp.tokenizer.add_special_case('%=', [{ORTH: '==', POS: SYM}])
    nlp.tokenizer.add_special_case('!=', [{ORTH: '==', POS: SYM}])
    nlp.tokenizer.add_special_case('<>', [{ORTH: '==', POS: SYM}])
