from typing import List


class Span:
    def __init__(self, start: int, end: int):
        self.start = start
        self.end = end

    def __str__(self):
        return f'[start={self.start}, end={self.end}'

    def __repr__(self):
        return str(self)


class Annotation:
    def __init__(self, filename: str, spans: List[Span]):
        self.spans = spans
        self.filename = filename

    def __str__(self):
        return f'filename={self.filename}, spans={self.spans}'

    def __repr__(self):
        return str(self)

    def strict_equal(self, another: 'Annotation'):
        if self.filename != another.filename:
            return False
        if len(self.spans) != len(another.spans):
            return False
        for s1, s2 in zip(self.spans, another.spans):
            if s1.start != s2.start:
                return False
            if s1.end != s2.end:
                return False
        return True

    def relaxed_equal(self, another: 'Annotation'):
        if self.filename != another.filename:
            return False
        for s1 in self.spans:
            for s2 in another.spans:
                if s2.start >= s1.end or s1.start >= s2.end:
                    continue
                return True
        return False