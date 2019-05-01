"""
Usage:
    eval_chemprot <gold> <pred>
"""
import docopt
import numpy as np
from blue import pmetrics
import pandas as pd


if __name__ == '__main__':
    argv = docopt.docopt(__doc__)

    labels = ["CPR:3", "CPR:4", "CPR:5", "CPR:6", "CPR:9", "false"]
    gold = []
    with open(argv['<gold>']) as fp:
        next(fp)
        for line in fp:
            toks = line.strip().split('\t')
            gold.append(labels.index(toks[-1]))

    df = pd.read_csv(argv['<pred>'], sep='\t', header=None)
    pred = []
    for i, row in df.iterrows():
        y = np.argmax(row.values)
        pred.append(y)

    result = pmetrics.classification_report(gold, pred, classes_=labels, macro=False, micro=True)
    subrst = result.sub_report([0, 1, 2, 3, 4], macro=False, micro=True)
    print(subrst.report)
