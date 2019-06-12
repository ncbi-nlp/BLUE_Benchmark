"""
Copyright (c) 2019, Yifan Peng
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this
  list of conditions and the following disclaimer in the documentation and/or
  other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import numpy as np
import pandas as pd
import tabulate
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support


class Report(object):
    def __init__(self):
        self.report = None
        self.table = None
        self.overall_acc = None
        self.kappa = None
        self.weighted_acc = None
        self.confusion = None

    def sensitivity(self, targetclass):
        return self.table.iloc[targetclass, 9]

    def specificity(self, targetclass):
        return self.table.iloc[targetclass, 10]

    def precision(self, targetclass):
        return self.table.iloc[targetclass, 5]

    def recall(self, targetclass):
        return self.table.iloc[targetclass, 6]

    def f1(self, targetclass):
        return self.table.iloc[targetclass, 7]

    def sub_report(self, targetclasses, *_, **kwargs) -> 'Report':
        digits = kwargs.pop('digits', 3)
        macro = kwargs.pop('macro', False)
        has_micro = kwargs.pop('micro', False)

        TP = np.zeros(len(targetclasses))
        TN = np.zeros(len(targetclasses))
        FP = np.zeros(len(targetclasses))
        FN = np.zeros(len(targetclasses))
        for i, targetclass in enumerate(targetclasses):
            TP[i] = self.table.iloc[targetclass, 1]
            TN[i] = self.table.iloc[targetclass, 2]
            FP[i] = self.table.iloc[targetclass, 3]
            FN[i] = self.table.iloc[targetclass, 4]

        TPR = tpr(TP, TN, FP, FN)
        TNR = tnr(TP, TN, FP, FN)
        PPV = ppv(TP, TN, FP, FN)
        NPV = npv(TP, TN, FP, FN)
        ACC = accuracy(TP, TN, FP, FN)
        F1 = f1(PPV, TPR)

        headings = ['Class', 'TP', 'TN', 'FP', 'FN',
                    'Precision', 'Recall', 'F-score',
                    'Accuracy', 'Sensitivity', 'Specificity', 'PPV', 'NPV',
                    'Support']
        tables = []

        for i, targetclass in enumerate(targetclasses):
            row = [t[i] for t in (TP, TN, FP, FN, PPV, TPR, F1, ACC, TPR, TNR, PPV, NPV, TP + FN)]
            tables.append([self.table.iloc[targetclass, 0]] + row)

        if has_micro:
            row = list(micro(TP, TN, FP, FN))
            tables.append(['micro'] + row)

        if macro:
            row = [np.nan] * 4
            row += [np.average(t) for t in [PPV, TPR, F1, ACC, TPR, TNR, PPV, NPV]]
            row += [np.nan]
            tables.append(['macro'] + row)

        df = pd.DataFrame(tables, columns=headings)
        float_formatter = ['g'] * 5 + ['.{}f'.format(digits)] * 8 + ['g']
        rtn = Report()
        rtn.report = tabulate.tabulate(df, showindex=False, headers=df.columns,
                                       tablefmt="plain", floatfmt=float_formatter)
        rtn.table = df
        rtn.overall_acc = overall_acc(TP, FN, FP, FN)
        rtn.weighted_acc = weighted_acc(TP, FN, FP, FN)
        return rtn


def divide(x, y):
    return np.true_divide(x, y, out=np.zeros_like(x, dtype=np.float), where=y != 0)


def tpr(tp, tn, fp, fn):
    """Sensitivity, hit rate, recall, or true positive rate"""
    return divide(tp, tp + fn)


def tnr(tp, tn, fp, fn):
    """Specificity or true negative rate"""
    return divide(tn, tn + fp)


def tp_tn_fp_fn(confusion_matrix):
    FP = np.sum(confusion_matrix, axis=0) - np.diag(confusion_matrix)
    FN = np.sum(confusion_matrix, axis=1) - np.diag(confusion_matrix)
    TP = np.diag(confusion_matrix)
    TN = np.sum(confusion_matrix) - (FP + FN + TP)
    return TP, TN, FP, FN


def ppv(tp, tn, fp, fn):
    """Precision or positive predictive value"""
    return divide(tp, tp + fp)


def npv(tp, tn, fp, fn):
    """Negative predictive value"""
    return divide(tn, tn + fn)


def fpr(tp, tn, fp, fn):
    """Fall out or false positive rate"""
    return divide(fp, fp + tn)


def fnr(tp, tn, fp, fn):
    """False negative rate"""
    return divide(fn, tp + fn)


def fdr(tp, tn, fp, fn):
    """False discovery rate"""
    return divide(fp, tp + fp)


def accuracy(tp, tn, fp, fn):
    """tp / N, same as """
    return divide(tp, tp + fn)


def f1(precision, recall):
    return divide(2 * precision * recall, precision + recall)


def cohen_kappa(confusion, weights=None):
    n_classes = confusion.shape[0]
    sum0 = np.sum(confusion, axis=0)
    sum1 = np.sum(confusion, axis=1)
    expected = divide(np.outer(sum0, sum1), np.sum(sum0))

    if weights is None:
        w_mat = np.ones([n_classes, n_classes], dtype=np.int)
        w_mat.flat[:: n_classes + 1] = 0
    elif weights == "linear" or weights == "quadratic":
        w_mat = np.zeros([n_classes, n_classes], dtype=np.int)
        w_mat += np.arange(n_classes)
        if weights == "linear":
            w_mat = np.abs(w_mat - w_mat.T)
        else:
            w_mat = (w_mat - w_mat.T) ** 2
    else:
        raise ValueError("Unknown kappa weighting type.")

    k = divide(np.sum(w_mat * confusion), np.sum(w_mat * expected))
    return 1 - k


def micro(tp, tn, fp, fn):
    """Returns tp, tn, fp, fn, ppv, tpr, f1, acc, tpr, tnr, ppv, npv, support"""
    TP, TN, FP, FN = [np.sum(t) for t in [tp, tn, fp, fn]]
    TPR = tpr(TP, TN, FP, FN)
    TNR = tnr(TP, TN, FP, FN)
    PPV = ppv(TP, TN, FP, FN)
    NPV = npv(TP, TN, FP, FN)
    FPR = fpr(TP, TN, FP, FN)
    FNR = fnr(TP, TN, FP, FN)
    FDR = fdr(TP, TN, FP, FN)
    F1 = f1(PPV, TPR)
    return TP, TN, FP, FN, PPV, TPR, F1, np.nan, TPR, TNR, PPV, NPV, TP + FN


def overall_acc(tp, tn, fp, fn):
    """Same as micro recall."""
    return divide(np.sum(tp), np.sum(tp + fn))


def weighted_acc(tp, tn, fp, fn):
    weights = tp + fn
    portion = divide(weights, np.sum(weights))
    acc = accuracy(tp, tn, fp, fn)
    return np.average(acc, weights=portion)


def micro_weighted(tp, tn, fp, fn):
    weights = tp + fn
    portion = divide(weights, np.sum(weights))
    # print(portion)
    TP, TN, FP, FN = [np.average(t, weights=portion) for t in [tp, tn, fp, fn]]
    TPR = tpr(TP, TN, FP, FN)
    TNR = tnr(TP, TN, FP, FN)
    PPV = ppv(TP, TN, FP, FN)
    NPV = npv(TP, TN, FP, FN)
    FPR = fpr(TP, TN, FP, FN)
    FNR = fnr(TP, TN, FP, FN)
    FDR = fdr(TP, TN, FP, FN)
    # ACC = accuracy(TP, TN, FP, FN)
    F1 = f1(PPV, TPR)
    return TP, TN, FP, FN, PPV, TPR, F1, np.nan, TPR, TNR, PPV, NPV, TP + FN


def confusion_matrix_report(confusion_matrix, *_, **kwargs) -> 'Report':
    classes_ = kwargs.get('classes_', None)
    digits = kwargs.pop('digits', 3)
    macro = kwargs.pop('macro', False)
    has_micro = kwargs.pop('micro', False)
    kappa_weights = kwargs.pop('kappa', None)

    TP, TN, FP, FN = tp_tn_fp_fn(confusion_matrix)
    TPR = tpr(TP, TN, FP, FN)
    TNR = tnr(TP, TN, FP, FN)
    PPV = ppv(TP, TN, FP, FN)
    NPV = npv(TP, TN, FP, FN)
    FPR = fpr(TP, TN, FP, FN)
    FNR = fnr(TP, TN, FP, FN)
    FDR = fdr(TP, TN, FP, FN)
    ACC = accuracy(TP, TN, FP, FN)
    F1 = f1(PPV, TPR)

    if classes_ is None:
        classes_ = [str(i) for i in range(confusion_matrix.shape[0])]

    headings = ['Class', 'TP', 'TN', 'FP', 'FN',
                'Precision', 'Recall', 'F-score',
                'Accuracy', 'Sensitivity', 'Specificity', 'PPV', 'NPV',
                'Support']
    tables = []

    for i, c in enumerate(classes_):
        row = [t[i] for t in (TP, TN, FP, FN, PPV, TPR, F1, ACC, TPR, TNR, PPV, NPV, TP + FN)]
        tables.append([str(c)] + row)

    if has_micro:
        row = list(micro(TP, TN, FP, FN))
        tables.append(['micro'] + row)

    if macro:
        row = [np.nan] * 4
        row += [np.average(t) for t in [PPV, TPR, F1, ACC, TPR, TNR, PPV, NPV]]
        row += [np.nan]
        tables.append(['macro'] + row)

    df = pd.DataFrame(tables, columns=headings)
    float_formatter = ['g'] * 5 + ['.{}f'.format(digits)] * 8 + ['g']
    rtn = Report()
    rtn.report = tabulate.tabulate(df, showindex=False, headers=df.columns,
                                   tablefmt="plain", floatfmt=float_formatter)
    rtn.table = df
    rtn.kappa = cohen_kappa(confusion_matrix, weights=kappa_weights)
    rtn.overall_acc = overall_acc(TP, FN, FP, FN)
    rtn.weighted_acc = weighted_acc(TP, FN, FP, FN)
    rtn.confusion = pd.DataFrame(confusion_matrix)
    return rtn


def auc(y_true, y_score, y_column: int = 1):
    """Compute Area Under the Curve (AUC).

    Args:
        y_true: (n_sample, )
        y_score: (n_sample, n_classes)
        y_column: column of y
    """
    fpr, tpr, _ = metrics.roc_curve(y_true, y_score[:, y_column], pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    return roc_auc, fpr, tpr


def multi_class_auc(y_true, y_score):
    """Compute Area Under the Curve (AUC).

    Args:
        y_true: (n_sample, n_classes)
        y_score: (n_sample, n_classes)
    """
    assert y_score.shape[1] == y_true.shape[1]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = y_score.shape[1]
    for i in range(n_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(y_true[:, i], y_score[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_true.ravel(), y_score.ravel())
    roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])
    return roc_auc, fpr, tpr


def classification_report(y_true, y_pred, *_, **kwargs) -> 'Report':
    """
    Args:
        y_true: (n_sample, )
        y_pred: (n_sample, )
    """
    m = metrics.confusion_matrix(y_true, y_pred)
    report = confusion_matrix_report(m, **kwargs)

    confusion = pd.DataFrame(m)
    if 'classes_' in kwargs:
        confusion.index = kwargs['classes_']
        confusion.columns = kwargs['classes_']
    report.confusion = confusion
    return report


def precision_recall_fscore_multilabel(y_true, y_pred, *_, **kwargs):
    """
    Args:
        y_true: (n_sample, n_classes)
        y_pred: (n_sample, n_classes)
    """
    example_based = kwargs.pop('example_based', False)
    if example_based:
        rs = []
        ps = []
        for yt, yp in zip(y_true, y_pred):
            p, r, _, _ = precision_recall_fscore_support(y_true=yt, y_pred=yp,
                                                         pos_label=1, average='binary')
            rs.append(r)
            ps.append(p)
        r = np.average(rs)
        p = np.average(ps)
        f1 = divide(2 * r * p, r + p)
    else:
        raise NotImplementedError
    return r, p, f1


"""
Test cases
"""


def test_cm1():
    cm = np.asarray([[20, 5], [10, 15]])
    k = cohen_kappa(cm)
    assert np.math.isclose(k, 0.4, rel_tol=1e-01)

    k = cohen_kappa(cm, weights='linear')
    assert np.math.isclose(k, 0.4, rel_tol=1e-01)


def test_cm2():
    cm = np.array([
        [236, 29, 7, 4, 8, 5, 3, 3, 1, 0, 5, 6, 1],
        [45, 3724, 547, 101, 102, 16, 0, 0, 2, 0, 0, 11, 0],
        [5, 251, 520, 132, 158, 11, 2, 1, 4, 0, 0, 4, 0],
        [0, 9, 71, 78, 63, 14, 2, 0, 0, 0, 0, 1, 0],
        [8, 37, 152, 144, 501, 200, 71, 11, 30, 3, 0, 18, 0],
        [5, 6, 6, 24, 144, 178, 136, 34, 30, 1, 0, 20, 0],
        [5, 2, 2, 3, 53, 115, 333, 106, 69, 4, 0, 36, 0],
        [2, 0, 0, 0, 1, 9, 99, 247, 119, 8, 0, 26, 0],
        [3, 2, 4, 7, 30, 54, 113, 124, 309, 78, 22, 72, 6],
        [1, 0, 0, 0, 1, 0, 2, 0, 5, 46, 17, 25, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 7, 53, 229, 28, 34],
        [18, 16, 5, 5, 16, 10, 11, 25, 29, 38, 70, 1202, 99],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 9, 11]
    ]).transpose()
    k = cohen_kappa(cm)
    assert np.math.isclose(k, 0.5547, rel_tol=1e-04)

    k = cohen_kappa(cm, weights='quadratic')
    assert np.math.isclose(k, 0.9214, rel_tol=1e-04)

    k = cohen_kappa(cm, weights='linear')
    assert np.math.isclose(k, 0.8332, rel_tol=1e-04)


def test_kappa():
    """
    Example 10.52

    Bernald Rosner, Fundamentals of Biostatistics (8th ed). Cengage Learning. 2016. p.434
    """
    cm = [[136, 92], [69, 240]]
    k = cohen_kappa(np.array(cm))
    assert np.math.isclose(k, 0.378, rel_tol=1e-03)


def test_precision_recall_fscore_multilabel():
    y_true = np.array([[0, 0, 1, 0]])
    y_pred = np.array([[0, 1, 1, 0]])
    r, p, f1 = precision_recall_fscore_multilabel(y_true, y_pred, example_based=True)
    assert r == 1
    assert p == 0.5

    y_true = np.array([[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1]])
    y_pred = np.array([[1, 0, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]])
    r, p, f1 = precision_recall_fscore_multilabel(y_true, y_pred, example_based=True)
    assert r == 1
    assert np.isclose(p, 0.888, 1e-02)


if __name__ == '__main__':
    test_precision_recall_fscore_multilabel()
