import csv
import itertools
import os
import re
from pathlib import Path
from typing import Match

import bioc
import fire
import pandas as pd
import tqdm

from blue.bert.create_chemprot_bert import print_rel_debug
from blue.bert.create_ddi_bert import replace_text

labels = ['PIP', 'TeCP', 'TeRP', 'TrAP', 'TrCP', 'TrIP', 'TrNAP', 'TrWP', 'false']


def read_text(pathname):
    with open(pathname) as fp:
        text = fp.read()
    sentences = []
    offset = 0
    for sent in text.split('\n'):
        sentence = bioc.BioCSentence()
        sentence.infons['filename'] = pathname.stem
        sentence.offset = offset
        sentence.text = sent
        sentences.append(sentence)
        i = 0
        for m in re.finditer('\S+', sent):
            if i == 0 and m.start() != 0:
                # add fake
                ann = bioc.BioCAnnotation()
                ann.id = f'a{i}'
                ann.text = ''
                ann.add_location(bioc.BioCLocation(offset, 0))
                sentence.add_annotation(ann)
                i += 1
            ann = bioc.BioCAnnotation()
            ann.id = f'a{i}'
            ann.text = m.group()
            ann.add_location(bioc.BioCLocation(m.start() + offset, len(m.group())))
            sentence.add_annotation(ann)
            i += 1
        offset += len(sent) + 1
    return sentences


def _get_ann_offset(sentences, match_obj: Match,
                    start_line_group, start_token_group,
                    end_line_group, end_token_group,
                    text_group):
    assert match_obj.group(start_line_group) == match_obj.group(end_line_group)
    sentence = sentences[int(match_obj.group(start_line_group)) - 1]

    start_token_idx = int(match_obj.group(start_token_group))
    end_token_idx = int(match_obj.group(end_token_group))
    start = sentence.annotations[start_token_idx].total_span.offset
    end = sentence.annotations[end_token_idx].total_span.end
    text = match_obj.group(text_group)

    actual = sentence.text[start - sentence.offset:end - sentence.offset].lower()
    expected = text.lower()
    assert actual == expected, 'Cannot match at %s:\n%s\n%s\nFind: %r, Matched: %r' \
                               % (
                               sentence.infons['filename'], sentence.text, match_obj.string, actual,
                               expected)
    return start, end, text


def read_annotations(pathname, sentences):
    anns = []
    pattern = re.compile(r'c="(.*?)" (\d+):(\d+) (\d+):(\d+)\|\|t="(.*?)"(\|\|a="(.*?)")?')
    with open(pathname) as fp:
        for i, line in enumerate(fp):
            line = line.strip()
            m = pattern.match(line)
            assert m is not None

            start, end, text = _get_ann_offset(sentences, m, 2, 3, 4, 5, 1)
            ann = {
                'start': start,
                'end': end,
                'type': m.group(6),
                'a': m.group(7),
                'text': text,
                'line': int(m.group(2)) - 1,
                'id': f'{pathname.name}.l{i}'
            }
            if len(m.groups()) == 9:
                ann['a'] = m.group(8)
            anns.append(ann)
    return anns


def _find_anns(anns, start, end):
    for ann in anns:
        if ann['start'] == start and ann['end'] == end:
            return ann
    raise ValueError


def read_relations(pathname, sentences, cons):
    pattern = re.compile(
        r'c="(.*?)" (\d+):(\d+) (\d+):(\d+)\|\|r="(.*?)"\|\|c="(.*?)" (\d+):(\d+) (\d+):(\d+)')

    relations = []
    with open(pathname) as fp:
        for line in fp:
            line = line.strip()
            m = pattern.match(line)
            assert m is not None

            start, end, text = _get_ann_offset(sentences, m, 2, 3, 4, 5, 1)
            ann1 = _find_anns(cons, start, end)
            start, end, text = _get_ann_offset(sentences, m, 8, 9, 10, 11, 7)
            ann2 = _find_anns(cons, start, end)
            relations.append({
                'docid': pathname.stem,
                'label': m.group(6),
                'Arg1': ann1['id'],
                'Arg2': ann2['id'],
                'string': line
            })
    return relations


def find_relations(relations, ann1, ann2):
    labels = []
    for i in range(len(relations) - 1, -1, -1):
        r = relations[i]
        if (r['Arg1'] == ann1['id'] and r['Arg2'] == ann2['id']) \
                or (r['Arg1'] == ann2['id'] and r['Arg2'] == ann1['id']):
            del relations[i]
            labels.append(r['label'])
    return labels


def convert(top_dir, dest):
    fp = open(dest, 'w')
    writer = csv.writer(fp, delimiter='\t', lineterminator='\n')
    writer.writerow(['index', 'sentence', 'label'])
    with os.scandir(top_dir / 'txt') as it:
        for entry in tqdm.tqdm(it):
            if not entry.name.endswith('.txt'):
                continue
            text_pathname = Path(entry.path)
            docid = text_pathname.stem

            sentences = read_text(text_pathname)
            # read assertions
            cons = read_annotations(top_dir / 'concept' / f'{text_pathname.stem}.con',
                                    sentences)
            # read relations
            relations = read_relations(top_dir / 'rel' / f'{text_pathname.stem}.rel',
                                       sentences, cons)
            for i, (con1, con2) in enumerate(itertools.combinations(cons, 2)):
                if con1['line'] != con2['line']:
                    continue
                # if con['type'] != 'treatment' or ast['type'] != 'problem':
                #     continue
                sentence = sentences[con1['line']]
                text = replace_text(sentence.text, sentence.offset, con1, con2)
                labels = find_relations(relations, con1, con2)
                if len(labels) == 0:
                    writer.writerow([f'{docid}.{con1["id"]}.{con2["id"]}', text, 'false'])
                else:
                    for l in labels:
                        writer.writerow([f'{docid}.{con1["id"]}.{con2["id"]}', text, l])

            if len(relations) != 0:
                for r in relations:
                    print(r['string'])
                    print_rel_debug(sentences, cons, r['Arg1'], r['Arg2'])
                    print('-' * 80)
    fp.close()


def split_doc(train1, train2, dev_docids, dest_dir):
    train1_df = pd.read_csv(train1, sep='\t')
    train2_df = pd.read_csv(train2, sep='\t')
    train_df = pd.concat([train1_df, train2_df])

    with open(dev_docids) as fp:
        dev_docids = fp.readlines()

    with open(dest_dir / 'train.tsv', 'w') as tfp, open(dest_dir / 'dev.tsv', 'w') as dfp:
        twriter = csv.writer(tfp, delimiter='\t', lineterminator='\n')
        twriter.writerow(['index', 'sentence', 'label'])
        dwriter = csv.writer(dfp, delimiter='\t', lineterminator='\n')
        dwriter.writerow(['index', 'sentence', 'label'])
        for i, row in train_df.iterrows():
            if row[0][:row[0].find('.')] in dev_docids:
                dwriter.writerow(row)
            else:
                twriter.writerow(row)


def create_i2b2_bert(gold_directory, output_directory):
    data_path = Path(gold_directory)
    dest_path = Path(output_directory)
    convert(data_path / 'original/reference_standard_for_test_data',
            dest_path / 'test.tsv')
    convert(data_path / 'original/concept_assertion_relation_training_data/beth',
            dest_path / 'train-beth.tsv')
    convert(data_path / 'original/concept_assertion_relation_training_data/partners',
            dest_path / 'train-partners.tsv')
    split_doc(dest_path / 'train-beth.tsv',
              dest_path / 'train-partners.tsv',
              data_path / 'dev-docids.txt',
              dest_path)


if __name__ == '__main__':
    fire.Fire(create_i2b2_bert)
