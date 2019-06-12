import json
from typing import List, Any, Dict


class Span:
    def __init__(self, start: int, end: int, text: str):
        self.start = start
        self.end = end
        self.text = text

    def __str__(self):
        return f'[start={self.start}, end={self.end}, text={self.text}]'

    def __repr__(self):
        return str(self)


class Annotation:
    def __init__(self, id: str, docid: str, spans: List[Span], type: str):
        self.spans = spans
        self.id = id
        self.docid = docid
        self.type = type

    def __str__(self):
        return f'docid={self.docid}, spans={self.spans}'

    def __repr__(self):
        return str(self)

    def strict_equal(self, another: 'Annotation') -> bool:
        if self.docid != another.docid:
            return False
        if len(self.spans) != len(another.spans):
            return False
        for s1, s2 in zip(self.spans, another.spans):
            if s1.start != s2.start:
                return False
            if s1.end != s2.end:
                return False
        return True

    def relaxed_equal(self, another: 'Annotation') -> bool:
        if self.docid != another.docid:
            return False
        for s1 in self.spans:
            for s2 in another.spans:
                if s2.start >= s1.end or s1.start >= s2.end:
                    continue
                return True
        return False

    def to_obj(self) -> Dict:
        return {
            'id': self.id,
            'docid': self.docid,
            'locations': [{'start': s.start, 'end': s.end, 'text': s.text} for s in self.spans],
            'type': self.type
        }

    @staticmethod
    def from_obj(obj: Any) -> "Annotation":
        return Annotation(obj['id'], obj['docid'],
                          [Span(o['start'], o['end'], o['text']) for o in obj['locations']],
                          obj['type'])


def read_annotations(pathname)->List[Annotation]:
    anns = []
    with open(pathname) as fp:
        for line in fp:
            obj = json.loads(line)
            anns.append(Annotation.from_obj(obj))
    return anns
