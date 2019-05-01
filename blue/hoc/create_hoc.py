"""
Usage:
    create <source_dir>

Options:
    source_dir  default: data/hoc
"""
import docopt
import tqdm
from pathlib import Path

from blue import pstring

LABELS = ['activating invasion and metastasis', 'avoiding immune destruction',
          'cellular energetics', 'enabling replicative immortality', 'evading growth suppressors',
          'genomic instability and mutation', 'inducing angiogenesis', 'resisting cell death',
          'sustaining proliferative signaling', 'tumor promoting inflammation']


def split_doc(docid_pathanme, data_dir, dest):
    with open(docid_pathanme) as fp:
        docids = [line.strip() for line in fp]

    lines = []
    for docid in tqdm.tqdm(docids):
        with open(data_dir / f'{docid}.txt', encoding='utf8') as fp:
            for i, line in enumerate(fp):
                idx = f'{docid}_s{i}'
                line = pstring.printable(line.strip())
                toks = line.strip().split('\t')
                text = toks[0]

                # read labels
                labels = set()
                for l in toks[1][1:-1].split(', '):
                    labels.add(l[1:-1])

                label_ids = []
                for i, label in enumerate(LABELS):
                    if label in labels:
                        label_ids.append('{}_{}'.format(i, 1))
                    else:
                        label_ids.append('{}_{}'.format(i, 0))
                lines.append((text, label_ids, idx))

    with open(dest, 'w') as fp:
        fp.write('labels\tsentence\tindex\n')
        for s in lines:
            fp.write('{}\t{}\t{}\n'.format(','.join(str(x) for x in s[1]), s[0], s[2]))


if __name__ == '__main__':
    argv = docopt.docopt(__doc__)
    top_dir = Path(argv['<source_dir>'])
    split_doc(top_dir / 'train_docid.txt',
              top_dir / 'original' / 'HoCCorpus',
              top_dir / 'train.tsv')
    split_doc(top_dir / 'dev_docid.txt',
              top_dir / 'original' / 'HoCCorpus',
              top_dir / 'dev.tsv')
    split_doc(top_dir / 'test_docid.txt',
              top_dir / 'original' / 'HoCCorpus',
              top_dir / 'test.tsv')
