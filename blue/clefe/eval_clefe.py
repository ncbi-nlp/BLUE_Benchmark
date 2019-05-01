"""
Usage:
    eval_clefe merge <gold_tsv> <pred_tsv> <pred_pubtator>
    eval_clefe eval <gold_directory> <pred_pubtator>

Options:
    <gold_directory>    Gold file folder. For example, Origin/Task1Gold_SN2012/Gold_SN2012
    <gold_pubtator>     Gold Pubtator file
    <gold_tsv>          Gold tsv file
    <pred_tsv>          Predicted tsv file
"""

import os
import tempfile
from pathlib import Path
from typing import List

import docopt
import tqdm

from blue import pmetrics
from blue.clefe.clefe_utils import Span, Annotation


def read_gold(dir) -> List[Annotation]:
    gold = []
    with os.scandir(dir) as it:
        for entry in tqdm.tqdm(it):
            with open(entry.path) as fp:
                for line in fp:
                    toks = line.strip().split('||')
                    filename = Path(toks[0]).stem
                    spans = []
                    for i in range(3, len(toks), 2):
                        spans.append(Span(int(toks[i]), int(toks[i + 1])))
                    gold.append(Annotation(filename, spans))
    return gold


def read_pubtator(pathname, entity_type) -> List[Annotation]:
    mentions = []
    with open(pathname) as fp:
        for line in fp:
            toks = line.strip().split('\t')
            if len(toks) >= 5 and toks[4] == entity_type:
                s = Span(int(toks[1]), int(toks[2]))
                mentions.append(Annotation(toks[0], [s]))
    return mentions


class Info:
    def __init__(self):
        self.spans = []
        self.filename = None


class State:
    def __init__(self, info: Info):
        self.info = info

    def run(self, toks, instances):
        raise NotImplementedError

    def next(self, label):
        raise NotImplementedError


class InitialState(State):
    def run(self, toks, instances):
        if self.info is not None:
            spans = []
            for i in range(0, len(self.info.spans), 2):
                spans.append(Span(self.info.spans[i], self.info.spans[i + 1]))
            instances.append(Annotation(self.info.filename, spans))
            self.info = None

    def next(self, label):
        if label == 'B' or label == 'I':
            return StateB(self.info)
        elif label == 'Empty' or label == 'O':
            return self
        else:
            raise ValueError


class StateB(State):
    def run(self, toks, instances):
        if self.info is not None:
            spans = []
            for i in range(0, len(self.info.spans), 2):
                spans.append(Span(self.info.spans[i], self.info.spans[i + 1]))
            instances.append(Annotation(self.info.filename, spans))
        self.info = Info()
        self.info.filename = toks[1]
        self.info.spans.append(int(toks[2]))
        self.info.spans.append(int(toks[2]) + len(toks[0]))

    def next(self, label):
        if label == 'B':
            return self
        elif label == 'I':
            return StateI(self.info)
        elif label == 'O':
            return StateO(self.info)
        elif label == 'Empty':
            return InitialState(self.info)
        else:
            raise ValueError


class StateI(State):
    def run(self, toks, instances):
        self.info.spans[-1] = int(toks[2]) + len(toks[0])

    def next(self, label):
        if label == 'B':
            return StateB(self.info)
        elif label == 'I':
            return self
        elif label == 'O':
            return StateO(self.info)
        elif label == 'Empty':
            return InitialState(self.info)
        else:
            raise ValueError


class StateO(State):
    def run(self, toks, instances):
        pass

    def next(self, label):
        if label == 'B':
            return StateB(self.info)
        elif label == 'I':
            return StateI2(self.info)
        elif label == 'O':
            return self
        elif label == 'Empty':
            return InitialState(self.info)
        else:
            raise ValueError


class StateI2(State):
    def run(self, toks, instances):
        self.info.spans.append(int(toks[2]))
        self.info.spans.append(int(toks[2]) + len(toks[0]))

    def next(self, label):
        if label == 'B':
            return StateB(self.info)
        elif label == 'I':
            return StateI(self.info)
        elif label == 'O':
            return StateO(self.info)
        elif label == 'Empty':
            return InitialState(self.info)
        else:
            raise ValueError


def read_pred(pathname) -> List[Annotation]:
    pred = []
    with open(pathname) as fp:
        state = InitialState(None)
        filename = None
        for i, line in enumerate(fp):
            line = line.strip()
            if len(line) == 0:
                toks = None
                filename = None
                label = 'Empty'
            else:
                toks = line.split('\t')
                assert len(toks) == 5, f'Error at {i}: \n{toks[1]}'
                label = toks[-1]
                if filename is None:
                    filename = toks[1]
                else:
                    if toks[1] != '-':
                        print(f'Error at {i}: \n{toks[1]}')
                        toks[1] = '-'
                    # assert toks[1] == '-', f'Error at {i}: \n{toks[1]}'
                    toks[1] = filename
            state = state.next(label)
            state.run(toks, pred)
        state = state.next('Empty')
        state.run(None, pred)
    return pred


def merge_pred(src1, src2, dest):
    def _read(pathname):
        with open(pathname) as fp:
            lines = [l.strip() for l in tqdm.tqdm(fp)]
            new_lines = []
            last_line = ''
            for l in lines:
                if len(l) == 0 and len(last_line) == 0:
                    continue
                new_lines.append(l)
                last_line = l
        return new_lines

    lines1 = _read(src1)
    lines2 = _read(src2)

    print('Lines1', len(lines1), 'Lines2', len(lines2))

    tmpfile = tempfile.mktemp()
    with open(tmpfile, 'w') as fout:
        for i in range(min(len(lines1), len(lines2))):
            line1 = lines1[i]
            line2 = lines2[i]

            if len(line1) == 0:
                assert len(line2) == 0, f'Error at {i}: \n{line1}\n{line2}'
                fout.write('\n')
                continue

            toks1 = line1.split('\t')
            toks2 = line2.split('\t')
            assert toks1[0] == toks2[0], f'Error at {i}: \n{line1}\n{line2}'
            # assert toks1[3] == toks2[1], f'Error at {i}: \n{line1}\n{line2}'
            fout.write(f'{line1}\t{toks2[-1]}\n')

    pred = read_pred(tmpfile)
    with open(dest, 'w') as fp:
        for mention in pred:
            fp.write('{}\t{}\t{}\t{}\t{}\n'.format(
                mention.filename, mention.spans[0].start, mention.spans[-1].end, '', 'Mention'
            ))

    return dest


def has_relaxed(target: Annotation, lst: List[Annotation]):
    for x in lst:
        if target.relaxed_equal(x):
            return True
    return False


def has_strict(target: Annotation, lst: List[Annotation]):
    for x in lst:
        if target.strict_equal(x):
            return True
    return False


def evaluate(gold, pred, has_function):
    # tp
    tps = []
    fns = []
    fps = []
    for g in gold:
        if has_function(g, pred):
            tps.append(g)
        else:
            fns.append(g)
    tps2 = []
    for p in pred:
        if has_function(p, gold):
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
    argv = docopt.docopt(__doc__)
    if argv['merge']:
        merge_pred(argv['<gold_tsv>'], argv['<pred_tsv>'], argv['<pred_pubtator>'])
    elif argv['eval']:
        gold = read_gold(argv['<gold_directory>'])
        print(gold[:10])
        pred = read_pubtator(argv['<pred_pubtator>'], 'Mention')
        evaluate(gold, pred, has_strict)

