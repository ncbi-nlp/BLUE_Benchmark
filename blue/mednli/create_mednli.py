"""
Usage:
    create_medni <jsonl> <dest>

Options:
    <jsonl>
    <dest>
"""

import csv
import json

import docopt
import tqdm

from blue import pstring

labels = ['contradiction', 'entailment', 'neutral']


def convert(src, dest):
    labels = set()
    with open(src, encoding='utf8') as fin, open(dest, 'w', encoding='utf8') as fout:
        writer = csv.writer(fout, delimiter='\t', lineterminator='\n')
        writer.writerow(['label', 'id', 'sentence1', 'sentence2'])
        for line in tqdm.tqdm(fin):
            line = pstring.printable(line, greeklish=True)
            obj = json.loads(line)
            writer.writerow([obj['gold_label'], obj['pairID'], obj['sentence1'], obj['sentence2']])
            labels.add(obj['gold_label'])
    print(sorted(labels))


if __name__ == '__main__':
    argv = docopt.docopt(__doc__)
    convert(argv['jsonl'], argv['dest'])
