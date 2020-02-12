import yaml
import urllib
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import numpy as np
import os

class BaseDataset(object):
    """Abstract dataset class"""

    def __init__(self, config_file):
        with open(config_file, encoding='utf8') as fp:
            self.config = yaml.load(fp)
        self.name = self.config['name'] if 'name' in self.config else ''
        self.description = self.config['description'] if 'description' in self.config else ''
        self.version = self.config['version'] if 'version' in self.config else ''
        self.citation = self.config['citation'] if 'citation' in self.config else ''
        self.links = self.config['links'] if 'links' in self.config else ''

    @property
    def full_name(self):
        """Full canonical name: (<name>_<version>)."""
        return '{}_{}'.format(self.name, self.version)

    def download(self, download_dir='blue_plus_data', override=False):
        """Downloads and prepares dataset for reading.

        Args:
          download_dir: string
            directory where downloaded files are stored.
            Defaults to "blue_plus_data/<full_name>".
          override: bool
            True to override the data
        Raises:
          IOError: if there is not enough disk space available.
        Returns:
          successful: bool
            True if download complete
        """
        raise NotImplementedError

    def evaluate(self, test_file, prediction_file, output_file):
        """Evaluate the predictions.

        Args:
          test_file: string
            location of the file containing the gold standards.
          prediction_file: string
            location of the file containing the predictions.
          output_file: string
            location of the file to store the evaluation results.
        Returns:
          results: string or pandas DataFrame that containing the evaluation results.
        """
        raise NotImplementedError


class form_tsv(object):
    def __init__(self, config_file="/rowdata/biosses/biosses.yml"):
        config_file = os.getcwd()+config_file
        with open(config_file, encoding='utf8') as fp:
            self.config = yaml.load(fp)
        self.save_path = self.config['save_path'] if 'save_path' in self.config else ''
        self.description = self.config['description'] if 'description' in self.config else ''
        self.path_pair = self.config['links'].get('path_pair') if 'links' in self.config else ""
        self.path_score = self.config['links'].get('path_score') if 'links' in self.config else ""
        self.data = None
    def biosses_format_data(self, col_index_pair=["index", "s1", 's2'],
                            col_index_score=['index', 'score1', 'score2', 'score3', 'score4', 'score5']):
        """

        self.path_pair: the path for pairs.xls
        self.path_score: the path for scores.xls
        :param col_index_pair: column index for pairs.xls
        :param col_index_score: column index for scores.xls
        :return: all formatted data in pandas frame
        """
        if self.path_score is '':
            raise FileNotFoundError("path_score doesn't exit,please check yml configuration file")
        if self.path_pair is '':
            raise FileNotFoundError("path_pair doesn't exit,please check yml configuration file")

        new_col_index = ['index', 'genre', 'filename', 'year', 'old_index', 'source1', 'source2', 'sentence1',
                         'sentence2', 'score']
        # drop the first line
        data = pd.read_excel(os.getcwd()+self.path_pair, header=0, drop=True)
        data.columns = col_index_pair
        index = [i for i in range(data.count()[0])]
        data['index'] = index

        score = pd.read_excel(os.getcwd()+self.path_score, header=0, drop=True)
        score.columns = col_index_score
        score['index'] = index
        score = score[score.columns[1:]].mean(axis=1)

        rtn = pd.DataFrame(index=index, columns=new_col_index)

        rtn['index'] = index
        rtn['gener'] = 'GENRE'
        rtn['filename'] = 'BIOSSES'
        rtn['year'] = 1997
        rtn['old_index'] = index
        rtn['source1'] = 'BIOSSES'
        rtn['source2'] = 'BIOSSES'
        rtn['sentence1'] = data[data.columns[1]]
        rtn['sentence2'] = data[data.columns[2]]
        rtn['score'] = score
        self.data = rtn
        return self.data

    def data_split(self, data, p_train=0.7, p_test=0.14):
        """

        :param data: all formatted data returned by biosses_format_data function
        :param p_train: percentage for training data
        :param p_test: percentage for testing data
        note: p_train + p_test < 1
        :return: train data, test data, dev data
        """
        assert p_train + p_test < 1, "the percentages of training and testing data should be less than 100%"
        data = data.sample(frac=1.0)

        data_count = data.count()
        train_count = np.floor(data_count * p_train)
        test_count = train_count + np.floor(data_count * p_test)

        data = data.reset_index()

        train = data.iloc[0:data_count]
        train = train.reset_index()

        test = data.iloc[data_count:test_count]
        test = test.reset_index()

        dev = data.iloc[test_count:]
        dev = dev.reset_index()

        test_results = test['score'].copy()
        return train, test, dev, test_results

    def save_files(self):
        data = self.biosses_format_data()
        splited_data = self.data_split(data,p_train= 0.7,p_test= 0.14)
        #train,test,dev,test_results

        for i,name in enumerate(self.save_path):
            path = self.save_path[name]
            if not os.path.exists(path):
                splited_data[i].to_csv(path_or_buf= path, sep='\t')
            else:
                print(path,"exits! ")
                break

fm = form_tsv()

fm.save_files()

