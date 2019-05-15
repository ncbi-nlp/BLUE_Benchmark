import fire
import pandas as pd
import numpy as np

from blue.ext.pmetrics import divide

LABELS = ['activating invasion and metastasis', 'avoiding immune destruction',
          'cellular energetics', 'enabling replicative immortality', 'evading growth suppressors',
          'genomic instability and mutation', 'inducing angiogenesis', 'resisting cell death',
          'sustaining proliferative signaling', 'tumor promoting inflammation']


def get_p_r_f_arrary(test_predict_label, test_true_label):
    num, cat = test_predict_label.shape
    acc_list = []
    prc_list = []
    rec_list = []
    f_score_list = []
    for i in range(num):
        label_pred_set = set()
        label_gold_set = set()

        for j in range(cat):
            if test_predict_label[i, j] == 1:
                label_pred_set.add(j)
            if test_true_label[i, j] == 1:
                label_gold_set.add(j)

        uni_set = label_gold_set.union(label_pred_set)
        intersec_set = label_gold_set.intersection(label_pred_set)

        tt = len(intersec_set)
        if len(label_pred_set) == 0:
            prc = 0
        else:
            prc = tt / len(label_pred_set)

        acc = tt / len(uni_set)

        rec = tt / len(label_gold_set)

        if prc == 0 and rec == 0:
            f_score = 0
        else:
            f_score = 2 * prc * rec / (prc + rec)

        acc_list.append(acc)
        prc_list.append(prc)
        rec_list.append(rec)
        f_score_list.append(f_score)

    mean_prc = np.mean(prc_list)
    mean_rec = np.mean(rec_list)
    f_score = divide(2 * mean_prc * mean_rec, (mean_prc + mean_rec))
    return mean_prc, mean_rec, f_score


def eval_hoc(true_file, pred_file):
    data = {}

    true_df = pd.read_csv(true_file, sep='\t')
    pred_df = pd.read_csv(pred_file, sep='\t')
    assert len(true_df) == len(pred_df), \
        f'Gold line no {len(true_df)} vs Prediction line no {len(pred_df)}'

    for i in range(len(true_df)):
        true_row = true_df.iloc[i]
        pred_row = pred_df.iloc[i]
        assert true_row['index'] == pred_row['index'], \
            'Index does not match @{}: {} vs {}'.format(i, true_row['index'], pred_row['index'])

        key = true_row['index'][:true_row['index'].find('_')]
        if key not in data:
            data[key] = (set(), set())

        if not pd.isna(true_row['labels']):
            for l in true_row['labels'].split(','):
                data[key][0].add(LABELS.index(l))

        if not pd.isna(pred_row['labels']):
            for l in pred_row['labels'].split(','):
                data[key][1].add(LABELS.index(l))

    assert len(data) == 315, 'There are 315 documents in the test set: %d' % len(data)

    y_test = []
    y_pred = []
    for k, (true, pred) in data.items():
        t = [0] * len(LABELS)
        for i in true:
            t[i] = 1

        p = [0] * len(LABELS)
        for i in pred:
            p[i] = 1

        y_test.append(t)
        y_pred.append(p)

    y_test = np.array(y_test)
    y_pred = np.array(y_pred)

    r, p, f1 = get_p_r_f_arrary(y_pred, y_test)
    print('Precision: {:.1f}'.format(p*100))
    print('Recall   : {:.1f}'.format(r*100))
    print('F1       : {:.1f}'.format(f1*100))


if __name__ == '__main__':
    fire.Fire(eval_hoc)
