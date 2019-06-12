import fire
import pandas as pd


def eval_sts(true_file, pred_file):
    true_df = pd.read_csv(true_file, sep='\t')
    pred_df = pd.read_csv(pred_file, sep='\t')
    assert len(true_df) == len(pred_df), \
        f'Gold line no {len(true_df)} vs Prediction line no {len(pred_df)}'
    for i in range(len(true_df)):
        true_row = true_df.iloc[i]
        pred_row = pred_df.iloc[i]
        assert true_row['index'] == pred_row['index'], \
            'Index does not match @{}: {} vs {}'.format(i, true_row['index'], pred_row['index'])
    print('Pearson correlation: {}'.format(true_df['score'].corr(pred_df['score'])))


if __name__ == '__main__':
    fire.Fire(eval_sts)
