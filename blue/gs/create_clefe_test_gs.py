import functools
import logging
import os
import re
from pathlib import Path

import jsonlines
import tqdm
import fire

from ext.data_structure import Span, Annotation


def pattern_repl(matchobj, prefix):
    """
    Replace [**Patterns**] with prefix+spaces.
    """
    s = matchobj.group(0).lower()
    return prefix.rjust(len(s))


def _proprocess_text(text):
    # noinspection PyTypeChecker
    text = re.sub(r'\[\*\*.*?\*\*\]', functools.partial(pattern_repl, prefix='PATTERN'),
                  text)
    # noinspection PyTypeChecker
    text = re.sub(r'(\|{4})|___|~~', functools.partial(pattern_repl, prefix=''), text)
    return text


def create_test_gs(reports_dir, anns_dir, output):
    anns_dir = Path(anns_dir)
    with jsonlines.open(output, 'w') as writer:
        with os.scandir(reports_dir) as it:
            for entry in tqdm.tqdm(it):
                text_file = Path(entry)
                with open(text_file) as fp:
                    text = fp.read()
                text = _proprocess_text(text)

                ann_file = anns_dir / text_file.name
                if not ann_file.exists():
                    logging.warning(f'{text_file.stem}: Cannot find ann file {ann_file}')
                    continue

                with open(ann_file) as fp:
                    for i, line in enumerate(fp):
                        line = line.strip()
                        toks = line.split('||')
                        type = toks[1]
                        spans = []
                        for i in range(3, len(toks), 2):
                            start = int(toks[i])
                            end = int(toks[i + 1])
                            spans.append(Span(start, end, text[start:end]))
                        a = Annotation(text_file.stem + f'.T{i}', text_file.stem, spans, type)
                        writer.write(a.to_obj())


if __name__ == '__main__':
    fire.Fire(create_test_gs)

