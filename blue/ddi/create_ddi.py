"""
Usage:
    create_medni <gold_directory> <dest>

Options:
    <gold_directory>
    <dest>
"""

import csv
import os
import re

import bioc
import docopt
from lxml import etree


def get_ann(arg, obj):
    for ann in obj['annotations']:
        if ann['id'] == arg:
            return ann
    raise ValueError


def replace_text(text, offset, ann1, ann2):
    ann1_start = ann1['start'] - offset
    ann2_start = ann2['start'] - offset
    ann1_end = ann1['end'] - offset
    ann2_end = ann2['end'] - offset

    if ann1_start <= ann2_start <= ann1_end \
            or ann1_start <= ann2_end <= ann1_end \
            or ann2_start <= ann1_start <= ann2_end \
            or ann2_start <= ann1_end <= ann2_end:
        start = min(ann1_start, ann2_start)
        end = max(ann1_end, ann2_end)
        before = text[:start]
        after = text[end:]
        return before + f'@{ann1["type"]}-{ann2["type"]}$' + after

    if ann1_start > ann2_start:
        ann1_start, ann1_end, ann2_start, ann2_end = ann2_start, ann2_end, ann1_start, ann1_end

    before = text[:ann1_start]
    middle = text[ann1_end:ann2_start]
    after = text[ann2_end:]

    return before + f'@{ann1["type"]}$' + middle + f'@{ann2["type"]}$' + after


def convert(top_dir, dest):
    fp = open(dest, 'w')
    writer = csv.writer(fp, delimiter='\t', lineterminator='\n')
    writer.writerow(['index', 'sentence', 'label'])
    cnt = 0
    for root, dirs, files in os.walk(top_dir):
        for name in files:
            pathname = os.path.join(root, name)
            tree = etree.parse(pathname)
            for stag in tree.xpath('/document/sentence'):
                sentence = bioc.BioCSentence()
                sentence.offset = 0
                sentence.text = stag.get('text')

                entities = {}
                for etag in stag.xpath('entity'):
                    m = re.match('(\d+)-(\d+)', etag.get('charOffset'))
                    assert m is not None
                    entities[etag.get('id')] = {
                        'start': int(m.group(1)),
                        'end': int(m.group(2)),
                        'type': etag.get('type'),
                        'id': etag.get('id'),
                        'text': etag.get('text')
                    }
                for rtag in stag.xpath('pair'):
                    if rtag.get('ddi') == 'false':
                        label = 'DDI-false'
                    else:
                        label = 'DDI-{}'.format(rtag.get('type'))
                        cnt += 1
                    e1 = entities.get(rtag.get('e1'))
                    e2 = entities.get(rtag.get('e2'))
                    text = replace_text(sentence.text, sentence.offset, e1, e2)
                    writer.writerow([f'{rtag.get("id")}', text, label])

    print(f'Have {cnt} relations')


if __name__ == '__main__':
    argv = docopt.docopt(__doc__)
    convert(argv['gold_directory'], argv['<dest>'])
