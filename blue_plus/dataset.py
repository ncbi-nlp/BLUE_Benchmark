import yaml


class BaseDataset(object):
    """Abstract dataset class"""

    def __init__(self, config_file):
        print(config_file)
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
