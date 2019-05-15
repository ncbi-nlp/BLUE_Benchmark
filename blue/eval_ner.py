from typing import List

import fire

from ext import pmetrics
from ext.data_structure import read_annotations, Annotation


def has_strict(target: Annotation, lst: List[Annotation]):
    for x in lst:
        if target.strict_equal(x):
            return True
    return False


def eval_cdr(gold_file, pred_file):
    golds = read_annotations(gold_file)
    preds = read_annotations(pred_file)

    # tp
    tps = []
    fns = []
    fps = []
    for g in golds:
        if has_strict(g, preds):
            tps.append(g)
        else:
            fns.append(g)
    tps2 = []
    for p in preds:
        if has_strict(p, golds):
            tps2.append(p)
        else:
            fps.append(p)

    tp = len(tps)
    fp = len(fps)
    fn = len(fns)
    tp2 = len(tps2)

    if tp != tp2:
        print(f'TP: {tp} vs TPs: {tp2}')

    TPR = pmetrics.tpr(tp, 0, fp, fn)
    PPV = pmetrics.ppv(tp, 0, fp, fn)
    F1 = pmetrics.f1(PPV, TPR)
    print('tp:      {}'.format(tp))
    print('fp:      {}'.format(fp))
    print('fn:      {}'.format(fn))
    print('pre:     {:.1f}'.format(PPV * 100))
    print('rec:     {:.1f}'.format(TPR * 100))
    print('f1:      {:.1f}'.format(F1 * 100))
    print('support: {}'.format(tp + fn))


if __name__ == '__main__':
    fire.Fire(eval_cdr)
