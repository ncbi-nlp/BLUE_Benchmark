import logging
import os
import sys
import urllib.request
from pathlib import Path

import pandas as pd
import yaml
from scipy.stats import pearsonr

sys.path.append(os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, os.pardir)))
from blue_plus.dataset import BaseDataset


class BIOSSES_Dataset(BaseDataset):
    def download(self, download_dir='blue_plus_data', override=False):
        download_dir = Path(download_dir)
        for local_name, url in self.links.items():
            local_data_path = download_dir / self.full_name / local_name
            data_exists = local_data_path.exists()
            if data_exists and not override:
                logging.info("Reusing dataset %s (%s)", self.name, local_data_path)
                continue
            logging.info('Downloading dataset %s (%s) to %s', self.name, url, local_data_path)
            urllib.request.urlretrieve(url, local_data_path)

    def evaluate(self, test_file, prediction_file, results_file):
        true_df = pd.read_csv(test_file, sep='\t')
        pred_df = pd.read_csv(prediction_file, sep='\t')
        assert len(true_df) == len(pred_df), \
            f'Gold line no {len(true_df)} vs Prediction line no {len(pred_df)}'

        p, _ = pearsonr(true_df['score'], pred_df['score'])
        print('Pearson: {:.3f}'.format(p))
        with open(results_file, 'w') as fp:
            fp.write('Pearson: {:.3f}'.format(p))

    def evaluate_bert(self, test_file, prediction_file, results_file):
        return self.evaluate(test_file, prediction_file, results_file)

    def prepare_bert_format(self, input_file, output_file):
        """Optional"""
        df = pd.read_csv(input_file, sep='\t')
        df = df['sentence1', 'sentence2', 'score']
        df.to_csv(output_file, sep='\t', index=None)


def main():
    logging.basicConfig(level=logging.INFO)
    dir = os.path.dirname(os.path.abspath(__file__))

    print("dir:",dir)

    d = BIOSSES_Dataset(os.path.join(dir, 'biosses.yml'))
    print('Name:       ', d.full_name)
    print('Description:', d.description)
    print('Citation:   ', d.citation)

    dir = Path('blue_plus_data') / d.full_name
    print("dir:",dir)
    dir.mkdir(parents=True, exist_ok=True)
    d.download(override=True)
    d.evaluate(dir / 'test.tsv', dir / 'test_results.tsv', dir / 'test_results.txt')


if __name__ == '__main__':
    main()
