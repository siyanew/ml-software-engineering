import torch
from nltk.translate.bleu_score import corpus_bleu as nltk_corpus_bleu, sentence_bleu as nltk_sentence_bleu
from torch import Tensor
from typing import List


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def sentence_bleu(hypothesis: Tensor, references: Tensor, k: int = 4) -> float:
    # Split the sentence
    hypo = hypothesis.split(" ")
    ref = references.split(" ")

    weights = [1 / k] * k

    return nltk_sentence_bleu(ref, hypo, weights)


def corpus_bleu(hypothesis: List[Tensor], references: List[Tensor], k: int = 4) -> float:
    # Split the sentences
    hypos = [h.split(" ") for h in hypothesis]
    refs = [r.split(" ") for r in references]
    weights = [1 / k] * k

    return nltk_corpus_bleu(refs, hypos, weights)
