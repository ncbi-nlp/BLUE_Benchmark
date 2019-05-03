"""
Usage:
    create [--hoc_dir=<dir>]

Options:
    --hoc_dir=<dir>     HoC directory [default: data/hoc]
"""
from pathlib import Path

import docopt
import tqdm


def split_doc(docid_pathanme, data_dir, dest):
    with open(docid_pathanme) as fp:
        docids = [line.strip() for line in fp]

    with open(dest, 'w', encoding='utf8') as fout:
        fout.write('index\tsentence\tlabels\n')

        for docid in tqdm.tqdm(docids):
            with open(data_dir / f'{docid}.txt', encoding='utf8') as fp:
                for i, line in enumerate(fp):
                    idx = f'{docid}_s{i}'
                    toks = line.strip().split('\t')
                    text = toks[0]
                    labels = set(l[1:-1] for l in toks[1][1:-1].split(', '))
                    labels = ','.join(sorted(labels))
                    fout.write(f'{idx}\t{text}\t{labels}\n')


if __name__ == '__main__':
    argv = docopt.docopt(__doc__)
    top_dir = Path(argv['--hoc_dir'])
    split_doc(top_dir / 'train_docid.txt',
              top_dir / 'original' / 'HoCCorpus',
              top_dir / 'train.tsv')
    split_doc(top_dir / 'dev_docid.txt',
              top_dir / 'original' / 'HoCCorpus',
              top_dir / 'dev.tsv')
    split_doc(top_dir / 'test_docid.txt',
              top_dir / 'original' / 'HoCCorpus',
              top_dir / 'test.tsv')
