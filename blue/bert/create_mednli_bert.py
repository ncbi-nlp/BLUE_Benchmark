import csv
import json

import fire
import tqdm
from pathlib import Path

from blue.ext import pstring


def convert(src, dest):
    with open(src, encoding='utf8') as fin, open(dest, 'w', encoding='utf8') as fout:
        writer = csv.writer(fout, delimiter='\t', lineterminator='\n')
        writer.writerow(['index', 'sentence1', 'sentence2', 'label'])
        for line in tqdm.tqdm(fin):
            line = pstring.printable(line, greeklish=True)
            obj = json.loads(line)
            writer.writerow([obj['pairID'], obj['sentence1'], obj['sentence2'], obj['gold_label']])


def create_mednli(input_dir, output_dir):
    mednli_dir = Path(input_dir)
    output_dir = Path(output_dir)
    for src_name, dst_name in zip(['mli_train_v1.jsonl', 'mli_dev_v1.jsonl', 'mli_test_v1.jsonl'],
                                  ['train.tsv', 'dev.tsv', 'test.tsv']):
        source = mednli_dir / src_name
        dest = output_dir / dst_name
        convert(source, dest)


if __name__ == '__main__':
    fire.Fire(create_mednli)
