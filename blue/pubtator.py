"""
Loads str/file-obj to a list of Pubtator objects
"""
import bioc
import logging
import re
from typing import List, Generator


class Pubtator:

    def __init__(self, pmid=None, title=None, abstract=None):
        self.pmid = pmid
        self.title = title
        self.abstract = abstract
        self.annotations = []
        self.relations = []
        self.has_disease = False
        self.has_gene = False
        self.has_mutation = False
        self.has_chemical = False

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

    def __str__(self):
        return '{self.pmid}\t{self.start}\t{self.end}\t{self.text}\t{self.type}\t{self.id}'.format(self=self)

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
    logger = logging.getLogger(__name__)
    doc = Pubtator()
    i = 0
    for i, line in enumerate(line_iterator, 1):
        if i % 100000 == 0:
            logger.debug('Read %d lines', i)
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
            annotation = PubtatorAnn(toks[0], int(toks[1]),
                                     int(toks[2]), toks[3],
                                     toks[4], toks[5])
            if annotation.type == 'Disease':
                doc.has_disease = True
            elif annotation.type == 'Gene':
                doc.has_gene = True
            elif annotation.type.endswith('Mutation') \
                    or annotation.type == 'SNP':
                doc.has_mutation = True
            elif annotation.type == 'Chemical':
                doc.has_chemical = True
            doc.annotations.append(annotation)
        if len(toks) == 4:
            relation = PubtatorRel(toks[0], toks[1], toks[2], toks[3])
            doc.relations.append(relation)

    if doc.pmid and (doc.title or doc.abstract):
        yield doc
    logger.debug('Read %d lines', i)


def iterparse(fp) -> Generator[Pubtator, None, None]:
    """
    Iteratively parse fp (file-like object) in pubtator format
    """
    return __iterparse(fp)


def load_annotation_s(s):
    """
    Parse s (a str) in the Pubtator annotation format
    """
    toks = s.split('\t')
    if len(toks) >= 6:
        return PubtatorAnn(pmid=toks[0], start=int(toks[1]),
                           end=int(toks[2]), text=toks[3],
                           type=toks[4], id=toks[5])
    return None


def dump2bioc(obj: Pubtator) -> bioc.BioCDocument:
    """
    Serialize obj (an instance of Pubtator) to a BioCDocument obj.

    Args:
        obj: a Pubtator document

    Return:
        a BioCDocument document
    """
    document = bioc.BioCDocument()
    document.id = obj.pmid
    title = bioc.BioCPassage()
    title.offset = 0
    title.text = obj.title
    document.add_passage(title)

    abstract = bioc.BioCPassage()
    abstract.offset = len(title.text) + 1
    abstract.text = obj.abstract
    document.add_passage(abstract)

    for annid, ann in enumerate(obj.annotations):
        annotation = bioc.BioCAnnotation()
        annotation.add_location(bioc.BioCLocation(ann.start, ann.end - ann.start))
        annotation.id = 'T{}'.format(annid)
        annotation.text = ann.text
        annotation.infons['type'] = ann.type
        annotation.infons['id'] = ann.id
        document.add_annotation(annotation)

    return document


def dumps(obj: Pubtator) -> str:
    """
    Serialize obj (an instance of Pubtator) to a Pubtator formatted str.

    Args:
        obj: a Pubtator document

    Return:
        a Pubtator formatted str
    """
    return str(obj)


def dump(fp, obj: Pubtator):
    """
    Serialize obj (an instance of Pubtator) to file-like object.

    Args:
        fp: a file-like object
        obj: a Pubtator document
    """
    fp.write(dumps(obj) + '\n')


def validate(obj):
    text = obj.title + '\n' + obj.abstract
    for ann in obj.annotations:
        try:
            actual = text[ann.start:ann.end]
            if actual != ann.text:
                return False
        except:
            return False
    return True


def merge(dst, *srcs):
    """
    Merge multiple Pubtator from srcs (file names) to dst
    """
    logger = logging.getLogger(__name__)
    docs = {}
    for src in srcs:
        logger.debug('Process %s', src)
        with open(src) as f:
            for doc in iterparse(f):
                if doc.pmid in docs:
                    # compare
                    if dumps(doc) != dumps(docs[doc.pmid]):
                        logging.warning('Two docs are not same\n%s\n\n%s',
                                        doc, docs[doc.pmid])
                else:
                    docs[doc.pmid] = doc

    with open(dst, 'w') as f:
        for pmid in sorted(docs.keys()):
            print(dumps(docs[pmid]), file=f)