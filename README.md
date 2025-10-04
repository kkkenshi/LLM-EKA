# LLM-EKA

This repository provides the source code for **NCRF++**, an open-source neural sequence labeling framework, extended for knowledge augmentation and entity-aware language model training.

**NCRF++ Repository:** [https://github.com/jiesutd/NCRFpp](https://github.com/jiesutd/NCRFpp)  

## Dataset

We use the **BioRED** and **METS-CoV** datasets for entity recognition and relation extraction tasks:

- BioRED GitHub: [https://github.com/ncbi/BioRED](https://github.com/ncbi/BioRED)
- METS-CoV GitHub[https://github.com/YLab-Open/METS-CoV](https://github.com/YLab-Open/METS-CoV)

## Knowledge Augmentation
`benchmark/YATO/generate_entities.py`

`benchmark/YATO/tools/convert_csv2bio.py`

`benchmark/YATO/generate_sentence.py`

## Training
`benchmark/YATO/run.py`




