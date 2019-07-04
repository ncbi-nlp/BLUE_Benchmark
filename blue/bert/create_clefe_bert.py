import functools
import os
import re
import shutil
from pathlib import Path

import fire
import tqdm
from lxml import etree

from blue.ext.preprocessing import tokenize_text, print_ner_debug, write_bert_ner_file


def pattern_repl(matchobj, prefix):
    """
    Replace [**Patterns**] with prefix+spaces.
    """
    s = matchobj.group(0).lower()
    return prefix.rjust(len(s))


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


def read_text(pathname):
    with open(pathname) as fp:
        text = fp.read()
    text = re.sub(r'\[\*\*.*?\*\*\]', functools.partial(pattern_repl, prefix='PATTERN'), text)
    text = re.sub(r'(\|{4})|___|~~', functools.partial(pattern_repl, prefix=''), text)
    sentences = tokenize_text(text, pathname.stem)

    # sentences = _cleanupSentences2(sentences)
    # sentences = _cleanupSentences1(sentences)
    # sentences = _normalize_sentences(sentences)
    # sentences = _tokenize_sentences(sentences)
    return sentences


def map_anns(sentences, ann_file):
    with open(ann_file) as fp:
        for line in fp:
            line = line.strip()
            toks = line.split('||')
            has_first = False
            for i in range(3, len(toks), 2):
                start = int(toks[i])
                end = int(toks[i + 1])
                anns = _find_toks(sentences, start, end)
                if len(anns) == 0:
                    print(f'Cannot find {ann_file}: {line}')
                    print_ner_debug(sentences, start, end)
                    exit(1)
                for ann in anns:
                    if not has_first:
                        ann.infons['NE_label'] = 'B'
                        has_first = True
                    else:
                        ann.infons['NE_label'] = 'I'
    return sentences


def convert(text_dir, ann_dir, dest, validate_mentions=None):
    total_sentences = []
    with os.scandir(text_dir) as it:
        for entry in tqdm.tqdm(it):
            text_file = Path(entry)
            ann_file = ann_dir / text_file.name
            if not ann_file.exists():
                print('Cannot find ann file:', ann_file)
                continue
            sentences = read_text(text_file)
            sentences = map_anns(sentences, ann_file)
            total_sentences.extend(sentences)

    # print(len(total_sentences))

    cnt = write_bert_ner_file(dest, total_sentences)
    if validate_mentions is not None and validate_mentions != cnt:
        print(f'Should have {validate_mentions}, but have {cnt} mentions')
    else:
        print(f'Have {cnt} mentions')


def convert_train_gs_to_text(src_dir, dest_dir):
    def _one_file(src_file, dest_file):
        # annotation
        with open(src_file) as fp:
            tree = etree.parse(fp)

        stringSlotMentions = {}
        for atag in tree.xpath('stringSlotMention'):
            stringSlotMentions[atag.get('id')] = atag.xpath('stringSlotMentionValue')[0].get(
                'value')

        classMentions = {}
        for atag in tree.xpath('classMention'):
            classMentions[atag.get('id')] = (atag.xpath('hasSlotMention')[0].get('id'),
                                             atag.xpath('mentionClass')[0].get('id'))

        with open(dest_file, 'w') as fp:
            for atag in tree.xpath('annotation'):
                id = atag.xpath('mention')[0].get('id')
                mentionClass = classMentions[id][1]
                try:
                    stringSlotMentionValue = stringSlotMentions[classMentions[id][0]]
                except:
                    stringSlotMentionValue = 'CUI-less'

                fp.write(f'{dest_file.name}||{mentionClass}||{stringSlotMentionValue}')
                for stag in atag.xpath('span'):
                    start = stag.get('start')
                    end = stag.get('end')
                    fp.write(f'||{start}||{end}')
                fp.write('\n')

    with os.scandir(src_dir) as it:
        for entry in tqdm.tqdm(it):
            path = Path(entry)
            basename = path.stem[:path.stem.find('.')]
            _one_file(path, dest_dir / f'{basename}.txt')


def split_development(data_path, devel_docids_pathname):
    with open(devel_docids_pathname) as fp:
        devel_docids = set(line.strip() for line in fp)

    os.mkdir(data_path / 'TRAIN_REPORTS')
    os.mkdir(data_path / 'DEV_REPORTS')

    with os.scandir(data_path / 'ALLREPORTS') as it:
        for entry in tqdm.tqdm(it):
            text_file = Path(entry)
            if text_file.stem in devel_docids:
                dest = data_path / 'DEV_REPORTS' / text_file.name
            else:
                dest = data_path / 'TRAIN_REPORTS' / text_file.name
            shutil.copy(text_file, dest)


def create_clefe_bert(gold_directory, output_directory):
    data_path = Path(gold_directory)
    dest_path = Path(output_directory)

    convert(data_path / 'Task1TrainSetCorpus199/TRAIN_REPORTS',
            data_path / 'Task1TrainSetGOLD199knowtatorehost/Task1Gold',
            dest_path / 'Training.tsv')

    convert(data_path / 'Task1TrainSetCorpus199/DEV_REPORTS',
            data_path / 'Task1TrainSetGOLD199knowtatorehost/Task1Gold',
            dest_path / 'Development.tsv')

    convert(data_path / 'Task1TestSetCorpus100/ALLREPORTS',
            data_path / 'Task1Gold_SN2012/Gold_SN2012',
            dest_path / 'Test.tsv')


if __name__ == '__main__':
    fire.Fire(create_clefe_bert)
