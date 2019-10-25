import os
import pathlib
import subprocess
import sys

import rouge


def calculate_bleu(ref_file, pred_file):
    pred = open(pred_file, mode='r')
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    bleu_script = pathlib.Path(scripts_dir, 'multi-bleu.perl')
    result = subprocess.run(['perl', str(bleu_script), str(ref_file)], stdin=pred,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(result.stdout.decode(sys.stdout.encoding))
    pred.close()


def calculate_rouge(ref_file: str, pred_file: str):
    ref = open(ref_file, mode='r')
    pred = open(pred_file, mode='r')

    evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'],
                            max_n=2,
                            length_limit_type='words',
                            alpha=0.5,  # Default F1_score
                            weight_factor=1.2,  # Default rouge score definition
                            stemming=True)

    references = ref.readlines()
    predictions = pred.readlines()

    scores = evaluator.get_scores(predictions, references)

    for name, score in scores.items():
        print('%-10s:' % (name + ":"), 'f: %-10f' % score['f'], 'p: %-10f' % score['p'], 'r: %-10f' % score['r'])


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print('Usage: python calculate_rouge.py <test_output_dir>')
        exit(1)

    path = pathlib.Path(sys.argv[1])
    ref_file = list(path.glob("*.msg"))[0]
    pred_file = list(path.glob("*.pred"))[0]

    calculate_bleu(ref_file, pred_file)
    calculate_rouge(ref_file, pred_file)
