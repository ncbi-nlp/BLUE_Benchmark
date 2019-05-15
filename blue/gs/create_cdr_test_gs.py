import fire
import jsonlines
import tqdm
import logging
from blue.ext import pubtator
from ext.data_structure import Span, Annotation


def create_test_gs(input, output, type):
    assert type in ('Chemical', 'Disease'), \
        'entity_type has to be Chemical or Disease'

    with open(input) as fp:
        docs = pubtator.load(fp)

    with jsonlines.open(output, 'w') as writer:
        for doc in tqdm.tqdm(docs):
            for i, ann in enumerate(doc.annotations):
                if ann.type != type:
                    continue
                expected_text = ann.text
                actual_text = doc.text[ann.start:ann.end]
                if expected_text != actual_text:
                    logging.warning('{}:{}: Text does not match. Expected {}. Actual {}'.format(
                        output, ann.line, repr(expected_text), repr(actual_text)))
                    continue
                a = Annotation(ann.pmid + f'.T{i}', doc.pmid,
                               [Span(ann.start, ann.end, ann.text)],
                               ann.type)
                writer.write(a.to_obj())


if __name__ == '__main__':
    fire.Fire(create_test_gs)
