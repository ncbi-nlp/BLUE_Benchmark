import csv
from pathlib import Path

import fire
import tqdm


def split_doc(docid_file, data_dir, dest):
    with open(docid_file) as fp:
        docids = [line.strip() for line in fp]

    with open(dest, 'w', encoding='utf8') as fout:
        writer = csv.writer(fout, delimiter='\t', lineterminator='\n')
        writer.writerow(['index', 'sentence1', 'sentence2', 'label'])

        for docid in tqdm.tqdm(docids):
            with open(data_dir / f'{docid}.txt', encoding='utf8') as fp:
                for i, line in enumerate(fp):
                    idx = f'{docid}_s{i}'
                    toks = line.strip().split('\t')
                    text = toks[0]
                    labels = set(l[1:-1] for l in toks[1][1:-1].split(', '))
                    labels = ','.join(sorted(labels))
                    writer.writerow([idx, text, labels])


def create_hoc(hoc_dir):
    hoc_dir = Path(hoc_dir)
    text_dir = hoc_dir / 'HoCCorpus'
    for name in ['train', 'dev', 'test']:
        print('Creating', name)
        split_doc(hoc_dir / f'{name}_docid.txt', text_dir, hoc_dir / f'{name}.tsv')


if __name__ == '__main__':
    fire.Fire(create_hoc)
