import json
import logging

import fire
import pandas as pd

from blue.ext import pmetrics

all_labels = set()


def _read_relations(pathname):
    objs = []
    df = pd.read_csv(pathname, sep='\t')
    for i, row in df.iterrows():
        obj = {'docid': row['docid'], 'id': row['id'], 'arg1': row['arg1'],
               'arg2': row['arg2'], 'label': row['label']}
        objs.append(obj)
        all_labels.add(obj['label'])
    return objs


def eval_chemprot(gold_file, pred_file):
    trues = _read_relations(gold_file)
    preds = _read_relations(pred_file)
    if len(trues) != len(preds):
        logging.error('%s-%s: Unmatched line no %s vs %s',
                      gold_file, pred_file, len(trues), len(preds))
        exit(1)

    labels = list(sorted(all_labels))

    y_test = []
    y_pred = []
    for i, (t, p) in enumerate(zip(trues, preds)):
        if t['docid'] != p['docid'] or t['arg1'] != p['arg1'] or t['arg2'] != p['arg2']:
            logging.warning('%s:%s-%s:%s: Cannot match %s vs %s',
                            gold_file, i, pred_file, i, t, p)
            continue
        y_test.append(labels.index(t['label']))
        y_pred.append(labels.index(p['label']))

    result = pmetrics.classification_report(y_test, y_pred, macro=False,
                                            micro=True, classes_=labels)
    print(result.report)
    print()

    subindex = [i for i in range(len(labels)) if labels[i] != 'false']
    result = result.sub_report(subindex, macro=False, micro=True)
    print(result.report)


if __name__ == '__main__':
    fire.Fire(eval_chemprot)
