import csv
import json

import fire
import tqdm

from blue.ext import pstring


def create_mednli_test_gs(input, output):
    with open(input, encoding='utf8') as fin, open(output, 'w', encoding='utf8') as fout:
        writer = csv.writer(fout, delimiter='\t', lineterminator='\n')
        writer.writerow(['index', 'label'])
        for line in tqdm.tqdm(fin):
            line = pstring.printable(line, greeklish=True)
            obj = json.loads(line)
            writer.writerow([obj['pairID'], obj['sentence1'], obj['sentence2'], obj['gold_label']])


if __name__ == '__main__':
    fire.Fire(create_mednli_test_gs)
