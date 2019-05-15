import fire
import tqdm

from blue.ext import pubtator
from blue.ext.preprocessing import tokenize_text, print_ner_debug, write_bert_ner_file


def _find_toks(sentences, start, end):
    toks = []
    for sentence in sentences:
        for ann in sentence.annotations:
            span = ann.total_span
            if start <= span.offset and span.offset + span.length <= end:
                toks.append(ann)
            elif span.offset <= start and end <= span.offset + span.length:
                toks.append(ann)
    return toks


def convert(src, dest, entity_type, validate_mentions=None):
    with open(src) as fp:
        docs = pubtator.load(fp)

    total_sentences = []
    for doc in tqdm.tqdm(docs):
        text = doc.title + ' ' + doc.abstract
        sentences = tokenize_text(text, doc.pmid)
        total_sentences.extend(sentences)

        for ann in doc.annotations:
            if ann.type == entity_type:
                anns = _find_toks(sentences, ann.start, ann.end)
                if len(anns) == 0:
                    print(f'Cannot find {doc.pmid}: {ann}')
                    print_ner_debug(sentences, ann.start, ann.end)
                    exit(1)
                has_first = False
                for ann in anns:
                    if not has_first:
                        ann.infons['NE_label'] = 'B'
                        has_first = True
                    else:
                        ann.infons['NE_label'] = 'I'

    cnt = write_bert_ner_file(dest, total_sentences)
    if validate_mentions is not None and validate_mentions != cnt:
        print(f'Should have {validate_mentions}, but have {cnt} {entity_type} mentions')
    else:
        print(f'Have {cnt} mentions')


class Commend:
    def chemical(self, input, output):
        convert(input, output, 'Chemical')

    def disease(self, input, output):
        convert(input, output, 'Disease')


if __name__ == '__main__':
    fire.Fire(Commend)
