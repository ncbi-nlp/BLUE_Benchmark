#!/usr/bin/env bash
python blue/gs/create_cdr_test_gs.py \
    --input data\BC5CDR\Original\CDR_TestSet.PubTator.txt \
    --output data\BC5CDR\CDR_TestSet.chem.jsonl \
    --type Chemical

python blue/gs/create_cdr_test_gs.py \
    --input data\BC5CDR\Original\CDR_TestSet.PubTator.txt \
    --output data\BC5CDR\CDR_TestSet.disease.jsonl \
    --type Disease

python blue/gs/create_clefe_test_gs.py \
    --reports_dir data\ShAReCLEFEHealthCorpus\Origin\Task1TestSetCorpus100\ALLREPORTS \
    --anns_dir data\ShAReCLEFEHealthCorpus\Origin\Task1Gold_SN2012\Gold_SN2012 \
    --output data\ShAReCLEFEHealthCorpus\Task1TestSetCorpus100_test_gs.jsonl

python blue/gs/create_chemprot_test_gs.py \
    --entities data\ChemProt\original\chemprot_test_gs\chemprot_test_entities_gs.tsv \
    --relations data\ChemProt\original\chemprot_test_gs\chemprot_test_gold_standard.tsv \
    --output data\ChemProt\chemprot_test_gs.tsv

python blue\gs\create_ddi_test_gs.py \
    --input_dir data\ddi2013-type\original\Test \
    --output data\ddi2013-type\test_gs.tsv

python blue\gs\create_i2b2_test_gs.py \
    --input_dir data\i2b2-2010\Original\reference_standard_for_test_data \
    --output_dir data\i2b2-2010\

python blue\gs\create_mednli_test_gs.py \
    --input data\mednli\Orignial\mli_test_v1.jsonl
    --output data\mednli\test_gs.tsv

# eval
python blue/eval_rel.py data\ChemProt\chemprot_test_gs.tsv data\ChemProt\chemprot_test_gs.tsv