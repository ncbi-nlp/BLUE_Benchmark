"""
Usage:
    eval_i2b2 <gold> <pred>
"""
import docopt
import numpy as np
import pandas as pd

from blue import pmetrics

if __name__ == '__main__':
    argv = docopt.docopt(__doc__)

    labels = ['PIP', 'TeCP', 'TeRP', 'TrAP', 'TrCP', 'TrIP', 'TrNAP', 'TrWP', 'false']
    df = pd.read_csv(argv['<gold>'], sep='\t')
    gold = []
    for i, row in df.iterrows():
        gold.append(labels.index(row.label))

    df = pd.read_csv(argv['<pred>'], sep='\t', header=None)
    pred = []
    for i, row in df.iterrows():
        y = np.argmax(row.values)
        pred.append(y)

    result = pmetrics.classification_report(gold, pred, classes_=labels, macro=False, micro=True)
    subrst = result.sub_report([0, 1, 2, 3, 4, 5, 6, 7], macro=False, micro=True)
    print(subrst.report)
