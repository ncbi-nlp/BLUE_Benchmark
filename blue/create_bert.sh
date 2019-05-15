#!/usr/bin/env bash
python blue/bert/create_mednli_bert.py data\mednli\Original data\mednli
python blue/bert/create_chemprot_bert.py data\ChemProt\original data\ChemProt\
python blue/bert/create_ddi_bert.py data\ddi2013-type\original\Test data\ddi2013-type\test.tsv