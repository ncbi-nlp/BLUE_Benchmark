# BLUE+ Protocol


## Task/Data description
Please provide a high-level description of your dataset to be included for BLUE+, with more details at your own web site. We also need a published reference for your dataset.


## Original data
Please prepare your dataset as follows, in order to be included in BLUE+:
* Each data instance should have a unique ID.
* The data needs to be split into training, validation, and test sets.


## Evaluation script
The evaluation script takes as input the test data and method output and generates detailed evaluation results. For example, for F1-metrics, TP, FP, FN, precision, and recall are required. Please provide your evaluation scripts to be included at BLUE+.


## Previous state-of-the-art (SOTA) results
The previous SOTA results should be provided from a published study, with a corresponding reference. The results should be verified by the evaluation script above.


## BERT Results
To be part of the benchmarking datasets, please report the results of your benchmarking task with the recent BERT model. Specifically, please provide:

 1. BERT-format files of the training, validation, and test sets
 2. BERT results
 3. Scripts to train and test BERT models. The scripts can be hosted at your own preferred repository. The scripts are available to users for model re-training and results verification.

If the data belongs to one of the tasks in BLUE (sentence similarity, named entity recognition, relation extraction, document classification, text inference), please follow the examples at NCBI_BERT ([https://github.com/ncbi-nlp/NCBI_BERT](https://github.com/ncbi-nlp/NCBI_BERT))


## Steps
 1. Pick a name for your dataset (e.g. CRAFT)
 2. Fork the BLUE_Benchmark project on GitHub ([https://github.com/ncbi-nlp/BLUE_Benchmark](https://github.com/ncbi-nlp/BLUE_Benchmark)), and create a new branch (e.g. craft)
 3. In your branch, create a subfolder (e.g., CRAFT) in the ‘blue_plus’ folder with at least the following files:
	 - your_dataset.yml - The configuration file
		 * Dataset name
		 * Dataset description
		 * Version
		 * The citation to use for this dataset
		 * Links to download the original data, BERT-formatted data and its results
		 * Your dataset license information
	 - your_dataset.py - downloading the datasets and evaluating the results.
		 * Implement a class to inherit the abstract class BaseDataset in [dataset.py](https://github.com/ncbi-nlp/BLUE_Benchmark/blob/master/blue_plus/dataset.py).
		 * The method `download` should download the data sets from the official internet distribution location (ie, “links” in the configuration)
		 * The method `evaluate` should evaluate the results
		 * CLI entry points to download the data
	- requirements.txt - a list of packages the script your_dataset.py relies on.
 4. Send a “pull request” back to BLUE-PLUS

An example dataset can be found at [https://github.com/ncbi-nlp/BLUE_Benchmark/tree/master/blue_plus](https://github.com/ncbi-nlp/BLUE_Benchmark/tree/master/blue_plus)

It may take up to 2-3 weeks to review your pull request. We may propose changes or request missing or additional information. Pull requests must be approved first before they can be merged. After the approval, we will include your dataset and results in the benchmark.
