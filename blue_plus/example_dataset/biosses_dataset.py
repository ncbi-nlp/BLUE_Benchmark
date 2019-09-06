import logging
import os
import urllib.request
from pathlib import Path

import pandas as pd
import yaml
from scipy.stats import pearsonr


class BIOSSES_Dataset:
    def __init__(self, config_file):
        print(config_file)
        with open(config_file, encoding='utf8') as fp:
            self.config = yaml.load(fp, yaml.FullLoader)
        self.name = self.config['name']
        self.description = self.config['description']
        self.version = self.config['version']
        self.citation = self.config['citation']
        self.links = self.config['links']

    @property
    def full_name(self):
        """Full canonical name: (<name>_<version>)."""
        return '{}_{}'.format(self.name, self.version)

    def download(self, download_dir='blue_plus_data', override=False):
        """Downloads and prepares dataset for reading.

        Args:
          download_dir: directory where downloaded files are stored.
            Defaults to "blue_plus_data/<full_name>".
          override: True to override the data
        Raises:
          IOError: if there is not enough disk space available.
        """
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
    d = BIOSSES_Dataset(os.path.join(dir, 'biosses.yml'))
    print('Name:       ', d.full_name)
    print('Description:', d.description)
    print('Citation:   ', d.citation)

    dir = Path('blue_plus_data') / d.full_name
    dir.mkdir(parents=True, exist_ok=True)
    d.download(override=True)
    d.evaluate(dir / 'test.tsv', dir / 'test_results.tsv', dir / 'test_results.txt')


if __name__ == '__main__':
    main()
