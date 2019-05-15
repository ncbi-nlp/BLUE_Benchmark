import collections
import csv
import itertools
from pathlib import Path

import fire
import tqdm

from blue.ext import pstring
from blue.ext.preprocessing import tokenize_text


def find_entities(sentence, entities, entity_type):
    es = []
    for e in entities:
        if e['type'] != entity_type:
            continue
        if sentence.offset <= e['start'] and e['end'] <= sentence.offset + len(sentence.text):
            es.append(e)
    return es


def find_relations(relations, chem, prot):
    labels = []
    for i in range(len(relations) - 1, -1, -1):
        r = relations[i]
        if r['Arg1'] == chem['id'] and r['Arg2'] == prot['id']:
            del relations[i]
            labels.append(r['label'])
    return labels


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
        return before + '@CHEM-GENE$' + after

    if ann1_start > ann2_start:
        ann1_start, ann1_end, ann2_start, ann2_end = ann2_start, ann2_end, ann1_start, ann1_end

    before = text[:ann1_start]
    middle = text[ann1_end:ann2_start]
    after = text[ann2_end:]

    if ann1['type'] in ('GENE-N', 'GENE-Y'):
        ann1['type'] = 'GENE'
    if ann2['type'] in ('GENE-N', 'GENE-Y'):
        ann2['type'] = 'GENE'

    return before + f'@{ann1["type"]}$' + middle + f'@{ann2["type"]}$' + after


def print_rel_debug(sentences, entities, id1, id2):
    e1 = None
    e2 = None
    for e in entities:
        if e['id'] == id1:
            e1 = e
        if e['id'] == id2:
            e2 = e
    assert e1 is not None and e2 is not None
    ss = [s for s in sentences
          if s.offset <= e1['start'] <= s.offset + len(s.text)
          or s.offset <= e2['start'] <= s.offset + len(s.text)]
    if len(ss) != 0:
        for s in ss:
            print(s.offset, s.text)
    else:
        for s in sentences:
            print(s.offset, s.text)


def merge_sentences(sentences):
    if len(sentences) == 0:
        return sentences

    new_sentences = []
    last_one = sentences[0]
    for s in sentences[1:]:
        if last_one.text[-1] in """.?!""" and last_one.text[-4:] != 'i.v.' and s.text[0].isupper():
            new_sentences.append(last_one)
            last_one = s
        else:
            last_one.text += ' ' * (s.offset - len(last_one.text) - last_one.offset)
            last_one.text += s.text
    new_sentences.append(last_one)
    return new_sentences


def convert(abstract_file, entities_file, relation_file, output):
    # abstract
    total_sentences = collections.OrderedDict()
    with open(abstract_file, encoding='utf8') as fp:
        for line in tqdm.tqdm(fp, desc=abstract_file.stem):
            toks = line.strip().split('\t')
            text = toks[1] + ' ' + toks[2]
            text = pstring.printable(text, greeklish=True)
            sentences = tokenize_text(text, toks[0])
            sentences = merge_sentences(sentences)
            total_sentences[toks[0]] = sentences
    # entities
    entities = collections.defaultdict(list)
    with open(entities_file, encoding='utf8') as fp:
        for line in tqdm.tqdm(fp, desc=entities_file.stem):
            toks = line.strip().split('\t')
            entities[toks[0]].append({
                'docid': toks[0],
                'start': int(toks[3]),
                'end': int(toks[4]),
                'type': toks[2],
                'id': toks[1],
                'text': toks[5]})
    # relations
    relations = collections.defaultdict(list)
    with open(relation_file, encoding='utf8') as fp:
        for line in tqdm.tqdm(fp, desc=relation_file.stem):
            toks = line.strip().split('\t')
            relations[toks[0]].append({
                'docid': toks[0],
                'label': toks[1],
                'Arg1': toks[2][toks[2].find(':') + 1:],
                'Arg2': toks[3][toks[3].find(':') + 1:],
                'toks': toks
            })

    with open(output, 'w') as fp:
        writer = csv.writer(fp, delimiter='\t', lineterminator='\n')
        writer.writerow(['index', 'sentence', 'label'])
        cnt = 0
        for docid, sentences in tqdm.tqdm(total_sentences.items(), total=len(total_sentences)):
            for sentence in sentences:
                # find chemical
                chemicals = find_entities(sentence, entities[docid], 'CHEMICAL')
                # find prot
                genes = find_entities(sentence, entities[docid], 'GENE-N') \
                        + find_entities(sentence, entities[docid], 'GENE-Y')
                for i, (chem, gene) in enumerate(itertools.product(chemicals, genes)):
                    text = replace_text(sentence.text, sentence.offset, chem, gene)
                    labels = find_relations(relations[docid], chem, gene)
                    if len(labels) == 0:
                        writer.writerow([f'{docid}.{chem["id"]}.{gene["id"]}', text, 'false'])
                    else:
                        for l in labels:
                            writer.writerow([f'{docid}.{chem["id"]}.{gene["id"]}', text, l])
                            cnt += 1

    # print('-' * 80)
    # for docid, rs in relations.items():
    #     if len(rs) > 0:
    #         for r in rs:
    #             print('\t'.join(r['toks']))
    #             print_rel_debug(total_sentences[r['docid']], entities[r['docid']],
    #                             r['Arg1'], r['Arg2'])
    #             print('-' * 80)


def create_chemprot_bert(data_dir, output_dir):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    convert(data_dir / 'chemprot_training/chemprot_training_abstracts.tsv',
            data_dir / 'chemprot_training/chemprot_training_entities.tsv',
            data_dir / 'chemprot_training/chemprot_training_gold_standard.tsv',
            output_dir / 'train.tsv')
    # convert(data_dir / 'chemprot_development/chemprot_development_abstracts.tsv',
    #         data_dir / 'chemprot_development/chemprot_development_entities.tsv',
    #         data_dir / 'chemprot_development/chemprot_development_gold_standard.tsv',
    #         output_dir / 'dev.tsv')
    # convert(data_dir / 'chemprot_test_gs/chemprot_test_abstracts_gs.tsv',
    #         data_dir / 'chemprot_test_gs/chemprot_test_entities_gs.tsv',
    #         data_dir / 'chemprot_test_gs/chemprot_test_gold_standard.tsv',
    #         output_dir / 'train.tsv')


if __name__ == '__main__':
    fire.Fire(create_chemprot_bert)
