import collections
import csv
import itertools
import os
import re
from pathlib import Path
from typing import Match

import bioc
import fire
import jsonlines
import tqdm

from ext.data_structure import Annotation, Span

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
                                   sentence.infons['filename'], sentence.text, match_obj.string,
                                   actual,
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


def create_test_gs(input_dir, output_dir):
    top_dir = Path(input_dir)
    dest = Path(output_dir)

    counter = collections.Counter()
    with jsonlines.open(dest / 'test_ann_gs.jsonl', 'w') as writer_ann, \
            open(dest / 'test_rel_gs.tsv', 'w') as fp_rel:
        writer_rel = csv.writer(fp_rel, delimiter='\t', lineterminator='\n')
        writer_rel.writerow(['id', 'docid', 'arg1', 'arg2', 'label'])
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
                for con in cons:
                    a = Annotation(con['id'], docid,
                                   [Span(con['start'], con['end'], con['text'])], con['type'])
                    writer_ann.write(a.to_obj())

                # read relations
                relations = read_relations(top_dir / 'rel' / f'{text_pathname.stem}.rel',
                                           sentences, cons)
                for i, (con1, con2) in enumerate(itertools.combinations(cons, 2)):
                    if con1['line'] != con2['line']:
                        continue
                    labels = find_relations(relations, con1, con2)
                    if len(labels) == 0:
                        writer_rel.writerow([f'{docid}.R{i}', docid, con1["id"], con2["id"],
                                             'false'])
                        counter['false'] += 1
                    else:
                        for l in labels:
                            writer_rel.writerow([f'{docid}.R{i}', docid, con1["id"], con2["id"], l])
                            counter[l] += 1
    for k, v in counter.items():
        print(k, v)


if __name__ == '__main__':
    fire.Fire(create_test_gs)
