import collections
import csv
import os
import re

import fire
from lxml import etree


def create_test_gs(input_dir, output):
    counter = collections.Counter()
    with open(output, 'w') as fp:
        writer = csv.writer(fp, delimiter='\t', lineterminator='\n')
        writer.writerow(['id', 'docid', 'arg1', 'arg2', 'label'])
        for root, dirs, files in os.walk(input_dir):
            for name in files:
                pathname = os.path.join(root, name)
                tree = etree.parse(pathname)
                docid = tree.xpath('/document')[0].get('id')
                for stag in tree.xpath('/document/sentence'):
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

                        e1 = entities.get(rtag.get('e1'))
                        e2 = entities.get(rtag.get('e2'))
                        writer.writerow([rtag.get("id"), docid, e1['id'], e2['id'], label])
                        counter[label] += 1
    for k, v in counter.items():
        print(k, v)


if __name__ == '__main__':
    fire.Fire(create_test_gs)
