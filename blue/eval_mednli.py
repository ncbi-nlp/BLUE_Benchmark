import fire
import pandas as pd

from blue.ext import pmetrics

labels = ['contradiction', 'entailment', 'neutral']


def eval_mednli(gold_file, pred_file):
    true_df = pd.read_csv(gold_file, sep='\t')
    pred_df = pd.read_csv(pred_file, sep='\t')
    assert len(true_df) == len(pred_df), \
        f'Gold line no {len(true_df)} vs Prediction line no {len(pred_df)}'

    y_test = []
    y_pred = []
    for i in range(len(true_df)):
        true_row = true_df.iloc[i]
        pred_row = pred_df.iloc[i]
        assert true_row['index'] == pred_row['index'], \
            'Index does not match @{}: {} vs {}'.format(i, true_row['index'], pred_row['index'])
        y_test.append(labels.index(true_row['label']))
        y_pred.append(labels.index(pred_row['label']))
    result = pmetrics.classification_report(y_test, y_pred, classes_=labels, macro=False, micro=True)
    print(result.report)


if __name__ == '__main__':
    fire.Fire(eval_mednli)
