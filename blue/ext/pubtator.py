"""
Loads str/file-obj to a list of Pubtator objects
"""
import logging
import re
from typing import List


class Pubtator:

    def __init__(self, pmid: str=None, title: str=None, abstract: str=None):
        self.pmid = pmid
        self.title = title
        self.abstract = abstract
        self.annotations = []  # type: List[PubtatorAnn]
        self.relations = []  # type: List[PubtatorRel]

    def __str__(self):
        text = self.pmid + '|t|' + self.title + '\n'
        if self.abstract:
            text += self.pmid + '|a|' + self.abstract + '\n'
        for ann in self.annotations:
            text += '{}\n'.format(ann)
        for rel in self.relations:
            text += '{}\n'.format(rel)
        return text

    def __iter__(self):
        yield 'pmid', self.pmid
        yield 'title', self.title
        yield 'abstract', self.abstract
        yield 'annotations', [dict(a) for a in self.annotations]
        yield 'relations', [dict(a) for a in self.relations]

    @property
    def text(self):
        """
        str: text
        """
        text = self.title
        if self.abstract:
            text += '\n' + self.abstract
        return text


class PubtatorAnn:
    def __init__(self, pmid, start, end, text, type, id):
        self.pmid = pmid
        self.start = start
        self.end = end
        self.text = text
        self.type = type
        self.id = id
        self.line = None

    def __str__(self):
        return f'{self.pmid}\t{self.start}\t{self.end}\t{self.text}\t{self.type}\t{self.id}'

    def __iter__(self):
        yield 'pmid', self.pmid
        yield 'start', self.start
        yield 'end', self.end
        yield 'text', self.text
        yield 'type', self.type
        yield 'id', self.id


class PubtatorRel:
    def __init__(self, pmid, type, id1, id2):
        self.pmid = pmid
        self.type = type
        self.id1 = id1
        self.id2 = id2
        self.line = None

    def __str__(self):
        return '{self.pmid}\t{self.type}\t{self.id1}\t{self.id2}'.format(self=self)

    def __iter__(self):
        yield 'pmid', self.pmid
        yield 'type', self.type
        yield 'id1', self.id1
        yield 'id2', self.id2


ABSTRACT_PATTERN = re.compile(r'(.*?)\|a\|(.*)')
TITLE_PATTERN = re.compile(r'(.*?)\|t\|(.*)')


def loads(s: str) -> List[Pubtator]:
    """
    Parse s (a str) to a list of Pubtator documents

    Returns:
        list: a list of PubTator documents
    """
    return list(__iterparse(s.splitlines()))


def load(fp) -> List[Pubtator]:
    """
    Parse file-like object to a list of Pubtator documents

    Args:
        fp: file-like object

    Returns:
        list: a list of PubTator documents
    """
    return loads(fp.read())


def __iterparse(line_iterator):
    """
    Iterative parse each line
    """
    doc = Pubtator()
    i = 0
    for i, line in enumerate(line_iterator, 1):
        if i % 100000 == 0:
            logging.debug('Read %d lines', i)
        line = line.strip()
        if not line:
            if doc.pmid and (doc.title or doc.abstract):
                yield doc
            doc = Pubtator()
            continue
        matcher = TITLE_PATTERN.match(line)
        if matcher:
            doc.pmid = matcher.group(1)
            doc.title = matcher.group(2)
            continue
        matcher = ABSTRACT_PATTERN.match(line)
        if matcher:
            doc.pmid = matcher.group(1)
            doc.abstract = matcher.group(2)
            continue
        toks = line.split('\t')
        if len(toks) >= 6:
            annotation = PubtatorAnn(toks[0], int(toks[1]), int(toks[2]), toks[3],
                                     toks[4], toks[5])
            annotation.line = i
            doc.annotations.append(annotation)
        if len(toks) == 4:
            relation = PubtatorRel(toks[0], toks[1], toks[2], toks[3])
            relation.line = i
            doc.relations.append(relation)

    if doc.pmid and (doc.title or doc.abstract):
        yield doc
    logging.debug('Read %d lines', i)
