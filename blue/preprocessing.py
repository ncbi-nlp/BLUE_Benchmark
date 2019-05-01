import csv
import re

import bioc
import en_core_web_sm

nlp = en_core_web_sm.load()


def split_punct(text, start):
    for m in re.finditer(r"""[\w']+|[!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~]""", text):
        yield m.group(), m.start() + start, m.end() + start


def tokenize_text(text, id):
    sentences = []
    doc = nlp(text)
    for sent in doc.sents:
        sentence = bioc.BioCSentence()
        sentence.infons['filename'] = id
        sentence.offset = sent.start_char
        sentence.text = text[sent.start_char:sent.end_char]
        sentences.append(sentence)
        i = 0
        for token in sent:
            for t, start, end in split_punct(token.text, token.idx):
                ann = bioc.BioCAnnotation()
                ann.id = f'a{i}'
                ann.text = t
                ann.add_location(bioc.BioCLocation(start, end-start))
                sentence.add_annotation(ann)
                i += 1
    return sentences


def print_ner_debug(sentences, start, end):
    anns = []
    for sentence in sentences:
        for ann in sentence.annotations:
            span = ann.total_span
            if start <= span.offset <= end \
                    or start <= span.offset + span.length <= end:
                anns.append(ann)
    print('-' * 80)
    if len(anns) != 0:
        for ann in anns:
            print(ann)
    print('-' * 80)
    ss = [s for s in sentences if s.offset <= start <= s.offset + len(s.text)]
    if len(ss) != 0:
        for s in ss:
            print(s.offset, s.text)
    else:
        for s in sentences:
            print(s.offset, s.text)


def write_bert_ner_file(dest, total_sentences):
    cnt = 0
    with open(dest, 'w') as fp:
        writer = csv.writer(fp, delimiter='\t', lineterminator='\n')
        for sentence in total_sentences:
            for i, ann in enumerate(sentence.annotations):
                if 'NE_label' not in ann.infons:
                    ann.infons['NE_label'] = 'O'
                elif ann.infons['NE_label'] == 'B':
                    cnt += 1
                if i == 0:
                    writer.writerow([ann.text, sentence.infons['filename'],
                                     ann.total_span.offset, ann.infons['NE_label']])
                else:
                    writer.writerow([ann.text, '-',
                                     ann.total_span.offset, ann.infons['NE_label']])
            fp.write('\n')
    return cnt


