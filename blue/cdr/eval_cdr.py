"""
Usage:
    eval_cdr eval -t type <gold_pubtator> <pred_pubtator>
    eval_cdr merge -t type <gold_tsv> <pred_tsv> <pred_pubtator>

Options:
    merge               create Putbator file from the tsv file
    <gold_pubtator>     Gold Pubtator file
    <gold_tsv>          Gold tsv file
    <pred_tsv>          Predicted tsv file
    -t type             Chemical or Disease
"""
import tempfile
from typing import List

import docopt
import tqdm

from blue.clefe.eval_clefe import has_strict, evaluate, read_pred, read_pubtator


def merge_pred(src1, src2, dest, type):
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
        line_number2 = 0
        for line_number1 in range(len(lines1)):
            if line_number2 >= len(lines2):
                break

            line1 = lines1[line_number1]
            line2 = lines2[line_number2]

            if len(line1) == 0 and len(line2) != 0:
                print(f'Error at {line_number1}: \n{line1}\n{line2}')
                continue

            if len(line1) == 0:
                assert len(line2) == 0, f'Error at {line_number1}: \n{line1}\n{line2}'
                fout.write('\n')
                continue

            while len(line2) == 0:
                line_number2 += 1
                line2 = lines2[line_number2]

            toks1 = line1.split()
            toks2 = line2.split()
            assert toks1[0] == toks2[0], \
                f'Error at {line_number1}: \n{line1}, {toks1[0]}\n{line2}, {toks2[0]}'
            # assert toks1[-1] == toks2[-1], \
            #     f'Error at {i}: \n{line1}, {toks1[-1]}\n{line2}, {toks1[-1]}'
            fout.write(f'{line1}\t{toks2[-1]}\n')
            line_number2 += 1

    pred = read_pred(tmpfile)
    with open(dest, 'w') as fp:
        for mention in pred:
            fp.write('{}\t{}\t{}\t{}\t{}\n'.format(
                mention.filename, mention.spans[0].start, mention.spans[-1].end, '', type
            ))

    return dest


if __name__ == '__main__':
    argv = docopt.docopt(__doc__)
    if argv['merge']:
        merge_pred(argv['<gold_tsv>'], argv['<pred_tsv>'], argv['<pred_pubtator>'], argv['-t'])
    elif argv['eval']:
        gold = read_pubtator(argv['<gold_pubtator>'], argv['-t'])
        pred = read_pubtator(argv['<pred_pubtator>'], argv['-t'])
        evaluate(gold, pred, has_strict)
