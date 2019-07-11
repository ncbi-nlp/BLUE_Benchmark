# BLUE, the Biomedical Language Understanding Evaluation benchmark

**\*\*\*\*\* New July 11th, 2019: preprocessed PubMed texts \*\*\*\*\***

We uploaded the [preprocessed PubMed texts](https://github.com/ncbi-nlp/NCBI_BERT/blob/master/README.md#pubmed)  that were used to pre-train the NCBI_BERT models.

**\*\*\*\*\* New June 17th, 2019: data in BERT format \*\*\*\*\***

We uploaded some [datasets](https://github.com/ncbi-nlp/BLUE_Benchmark/releases/tag/0.1) that are ready to be used with the [NCBI_BERT codes](https://github.com/ncbi-nlp/NCBI_BERT).

## Introduction

BLUE benchmark consists of five different biomedicine text-mining tasks with ten corpora.
Here, we rely on preexisting datasets because they have been widely used by the BioNLP community as shared tasks.
These tasks cover a diverse range of text genres (biomedical literature and clinical notes), dataset sizes, and degrees of difficulty and, more importantly, highlight common biomedicine text-mining challenges.

## Tasks

| Corpus          | Train |  Dev | Test | Task                    | Metrics             | Domain     |
|-----------------|------:|-----:|-----:|-------------------------|---------------------|------------|
| MedSTS          |   675 |   75 |  318 | Sentence similarity     | Pearson             | Clinical   |
| BIOSSES         |    64 |   16 |   20 | Sentence similarity     | Pearson             | Biomedical |
| BC5CDR-disease  |  4182 | 4244 | 4424 | NER                     | F1                  | Biomedical |
| BC5CDR-chemical |  5203 | 5347 | 5385 | NER                     | F1                  | Biomedical |
| ShARe/CLEFE     |  4628 | 1075 | 5195 | NER                     | F1                  | Clinical   |
| DDI             |  2937 | 1004 |  979 | Relation extraction     | micro F1            | Biomedical |
| ChemProt        |  4154 | 2416 | 3458 | Relation extraction     | micro F1            | Biomedical |
| i2b2-2010       |  3110 |   11 | 6293 | Relation extraction     | F1                  | Clinical   |
| HoC             |  1108 |  157 |  315 | Document classification | F1                  | Biomedical |
| MedNLI          | 11232 | 1395 | 1422 | Inference               | accuracy            | Clinical   |


### Sentence similarity

[BIOSSES](http://tabilab.cmpe.boun.edu.tr/BIOSSES/) is a corpus of sentence pairs selected from the Biomedical Summarization Track Training Dataset in the biomedical domain.
Here, we randomly select 80% for training and 20% for testing because there is no standard splits in the released data.

[MedSTS](https://mayoclinic.pure.elsevier.com/en/publications/medsts-a-resource-for-clinical-semantic-textual-similarity) is a corpus of sentence pairs selected from Mayo Clinics clinical data warehouse.
Please visit the website to obtain a copy of the dataset.
We use the standard training and testing sets in the shared task.

### Named entity recognition

[BC5CDR](https://biocreative.bioinformatics.udel.edu/tasks/biocreative-v/track-3-cdr/) is a collection of 1,500 PubMed titles and abstracts selected from the CTD-Pfizer corpus and was used in the BioCreative V chemical-disease relation task
We use the standard training and test set in the BC5CDR shared task

[ShARe/CLEF](https://physionet.org/works/ShAReCLEFeHealth2013/) eHealth Task 1 Corpus is a collection of 299 deidentified clinical free-text notes from the MIMIC II database
Please visit the website to obtain a copy of the dataset.
We use the standard training and test set in the ShARe/CLEF eHealth Tasks 1.

### Relation extraction

[DDI](http://labda.inf.uc3m.es/ddicorpus) extraction 2013 corpus is a collection of 792 texts selected from the DrugBank database and other 233 Medline abstracts
In our benchmark, we use 624 train files and 191 test files to evaluate the performance and report the micro-average F1-score of the four DDI types.

[ChemProt](https://biocreative.bioinformatics.udel.edu/news/corpora/) consists of 1,820 PubMed abstracts with chemical-protein interactions and was used in the BioCreative VI text mining chemical-protein interactions shared task
We use the standard training and test sets in the ChemProt shared task and evaluate the same five classes: CPR:3, CPR:4, CPR:5, CPR:6, and CPR:9.

### Document multilabel classification

[HoC](https://www.cl.cam.ac.uk/~sb895/HoC.html) (the Hallmarks of Cancers corpus) consists of 1,580 PubMed abstracts annotated with ten currently known hallmarks of cancer
We use 315 (~20%) abstracts for testing and the remaining abstracts for training. For the HoC task, we followed the common practice and reported the example-based F1-score on the abstract level

### Inference task

[MedNLI](https://physionet.org/physiotools/mimic-code/mednli/) is a collection of sentence pairs selected from MIMIC-III. We use the same training, development,
and test sets in [Romanov and Shivade](https://www.aclweb.org/anthology/D18-1187)

### Datasets

Some datasets can be downloaded at [https://github.com/ncbi-nlp/BLUE_Benchmark/releases/tag/0.1](https://github.com/ncbi-nlp/BLUE_Benchmark/releases/tag/0.1)

## Baselines

| Corpus          | Metrics | SOTA* | ELMo | BioBERT | NCBI_BERT(base) (P) | NCBI_BERT(base) (P+M) | NCBI_BERT(large) (P) | NCBI_BERT(large) (P+M) |
|-----------------|--------:|------:|-----:|--------:|--------------------:|----------------------:|---------------------:|-----------------------:|
| MedSTS          | Pearson |  83.6 | 68.6 |    84.5 |                84.5 |                  84.8 |                 84.6 |                   83.2 |
| BIOSSES         | Pearson |  84.8 | 60.2 |    82.7 |                89.3 |                  91.6 |                 86.3 |                   75.1 |
| BC5CDR-disease  |       F |  84.1 | 83.9 |    85.9 |                86.6 |                  85.4 |                 82.9 |                   83.8 |
| BC5CDR-chemical |       F |  93.3 | 91.5 |    93.0 |                93.5 |                  92.4 |                 91.7 |                   91.1 |
| ShARe/CLEFE     |       F |  70.0 | 75.6 |    72.8 |                75.4 |                  77.1 |                 72.7 |                   74.4 |
| DDI             |       F |  72.9 | 78.9 |    78.8 |                78.1 |                  79.4 |                 79.9 |                   76.3 |
| ChemProt        |       F |  64.1 | 66.6 |    71.3 |                72.5 |                  69.2 |                 74.4 |                   65.1 |
| i2b2            |       F |  73.7 | 71.2 |    72.2 |                74.4 |                  76.4 |                 73.3 |                   73.9 |
| HoC             |       F |  81.5 | 80.0 |    82.9 |                85.3 |                  83.1 |                 87.3 |                   85.3 |
| MedNLI          |     acc |  73.5 | 71.4 |    80.5 |                82.2 |                  84.0 |                 81.5 |                   83.8 |

**P**: PubMed, **P+M**: PubMed + MIMIC-III

SOTA, state-of-the-art as of April 2019, to the best of our knowledge

* **MedSTS, BIOSSES**: Chen et al. 2019. [BioSentVec: creating sentence embeddings for biomedical texts](https://arxiv.org/abs/1810.09302v2). In Proceedings of the 7th IEEE International Conference on Healthcare Informatics.
* **BC5CDR-disease, BC5CDR-chem**: Yoon et al. 2018. [CollaboNet: collaboration of deep neural networks for biomedical named entity recognition](https://arxiv.org/abs/1809.07950v1). arXiv preprint arXiv:1809.07950.
* **ShARe/CLEFE**: Leaman et al. 2015. [Challenges in clinical natural language processing for automated disorder normalization](https://www.sciencedirect.com/science/article/pii/S1532046415001501?via%3Dihub). Journal of biomedical informatics, 57:28–37.
* **DDI**: Zhang et al. 2018. [Drug-drug interaction extraction via hierarchical RNNs on sequence and shortest dependency paths](https://academic.oup.com/bioinformatics/article/34/5/828/4565590). Bioinformatics (Oxford, England), 34:828–835.
* **Chem-Prot**: Peng et al. 2018. [Extracting chemical-protein relations with ensembles of SVM and deep learning models](https://academic.oup.com/database/article/doi/10.1093/database/bay073/5055578). Database: the journal of biological
databases and curation, 2018.
* **i2b2**: Rink et al. 2011. [Automatic extraction of relations between medical concepts in clinical texts](https://academic.oup.com/jamia/article/18/5/594/833364). Journal of the American Medical Informatics Association, 18:594–600.
* **HoC**: Du et al. 2019. [ML-Net: multilabel classification of biomedical texts with deep neural networks](https://arxiv.org/abs/1811.05475v2). Journal of the American Medical Informatics Association (JAMIA).
* **MedNLI**: Romanov et al. 2018. [Lessons from natural language inference in the clinical domain](https://www.aclweb.org/anthology/D18-1187). In Proceedings of EMNLP, pages 1586–1596.



### Fine-tuning with ELMo

We adopted the ELMo model pre-trained on PubMed abstracts to accomplish the BLUE tasks.
The output of ELMo embeddings of each token is used as input for the fine-tuning model. 
We retrieved the output states of both layers in ELMo and concatenated them into one vector for each word. We used the maximum sequence length 128 for padding. 
The learning rate was set to 0.001 with an Adam optimizer.
We iterated the training process for 20 epochs with batch size 64 and early stopped if the training loss did not decrease.

### Fine-tuning with BERT

Please see [https://github.com/ncbi-nlp/NCBI_BERT](https://github.com/ncbi-nlp/NCBI_BERT).


## Citing BLUE

*  Peng Y, Yan S, Lu Z. [Transfer Learning in Biomedical Natural Language Processing: An
Evaluation of BERT and ELMo on Ten Benchmarking Datasets](https://arxiv.org/abs/1906.05474). In *Proceedings of the Workshop on Biomedical Natural Language Processing (BioNLP)*. 2019.

```
@InProceedings{peng2019transfer,
  author    = {Yifan Peng and Shankai Yan and Zhiyong Lu},
  title     = {Transfer Learning in Biomedical Natural Language Processing: An Evaluation of BERT and ELMo on Ten Benchmarking Datasets},
  booktitle = {Proceedings of the 2019 Workshop on Biomedical Natural Language Processing (BioNLP 2019)},
  year      = {2019},
}
```

## Acknowledgments

This work was supported by the Intramural Research Programs of the National Institutes of Health, National Library of
Medicine and Clinical Center. This work was supported by the National Library of Medicine of the National Institutes of Health under award number K99LM013001-01.

We are also grateful to the authors of BERT and ELMo to make the data and codes publicly available.

## Disclaimer

This tool shows the results of research conducted in the Computational Biology Branch, NCBI. The information produced
on this website is not intended for direct diagnostic use or medical decision-making without review and oversight
by a clinical professional. Individuals should not change their health behavior solely on the basis of information
produced on this website. NIH does not independently verify the validity or utility of the information produced
by this tool. If you have questions about the information produced on this website, please see a health care
professional. More information about NCBI's disclaimer policy is available.
