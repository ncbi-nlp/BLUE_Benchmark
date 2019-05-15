import collections
import csv
from pathlib import Path

import fire
import jsonlines


def _read_entities(pathname):
    d = collections.defaultdict(list)
    with open(pathname, encoding='utf8') as fp:
        for line in fp:
            toks = line.strip().split()
            d[toks[0]].append(toks)
    return d


def _read_relations(pathname):
    d = collections.defaultdict(list)
    with open(pathname, encoding='utf8') as fp:
        for line in fp:
            toks = line.strip().split()
            arg1 = toks[2][toks[2].find(':') + 1:]
            arg2 = toks[3][toks[3].find(':') + 1:]
            d[toks[0], arg1, arg2].append(toks)
    return d


def create_test_gs(entities, relations, output):
    entities = _read_entities(entities)
    relations = _read_relations(relations)

    counter = collections.Counter()
    with open(output, 'w') as fp:
        writer = csv.writer(fp, delimiter='\t', lineterminator='\n')
        writer.writerow(['id', 'docid', 'arg1', 'arg2', 'label'])
        for docid, ents in entities.items():
            chemicals = [e for e in ents if e[2] == 'CHEMICAL']
            genes = [e for e in ents if e[2] != 'CHEMICAL']
            i = 0
            for c in chemicals:
                for g in genes:
                    k = (docid, c[1], g[1])
                    if k in relations:
                        for l in relations[k]:
                            label = l[1]
                            writer.writerow([f'{docid}.R{i}', docid, k[1], k[2], label])
                            counter[label] += 1
                            i += 1
                    else:
                        writer.writerow([f'{docid}.R{i}', docid, k[1], k[2], 'false'])
                        i += 1
    for k, v in counter.items():
        print(k, v)


if __name__ == '__main__':
    fire.Fire(create_test_gs)
