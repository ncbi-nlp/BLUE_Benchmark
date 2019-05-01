"""
Usage:
    eval_mednli <gold> <pred>
"""
import docopt
import numpy as np
import pandas as pd

from blue import pmetrics

if __name__ == '__main__':
    argv = docopt.docopt(__doc__)

    labels = ['contradiction', 'entailment', 'neutral']

    gold = []
    with open(argv['<gold>']) as fp:
        next(fp)
        for line in fp:
            toks = line.strip().split('\t')
            gold.append(labels.index(toks[0]))

    df = pd.read_csv(argv['<pred>'], sep='\t', header=None)
    pred = []
    for i, row in df.iterrows():
        y = np.argmax(row.values)
        pred.append(y)

    result = pmetrics.classification_report(gold, pred, classes_=labels, macro=False, micro=True)
    print('acc', result.overall_acc)
    print(result.report)
