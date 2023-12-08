# About
Contains datasets and code for the paper "Error types in Transformer-based Paraphrasing Models: A Taxonomy, Paraphrase Annotation Model and Dataset". 

## Table of Contents
- [SNIPS Folder](#snips-folder)
- [Generated Paraphrases](#generated-paraphrases)
- [Appendices](#appendices)
- [TPME Dataset](#tpme-dataset)
- [Possible use cases](#possible-use-cases)
- [More information](#more-information)

# SNIPS Folder
This folder contains the code to extract seed utterances from raw SNIPS JSON snippets retrieved from the [SNIPS][snips] dataset (described in Section 2.2).

# Generated Paraphrases
This folder contains the generated paraphrases described in Section 2.4 of our paper. There are 22864 generated paraphrases for 598 seed utterances. The generated paraphrases are grouped into 8 datasets based on the model used to generate them and the intent of the seed utterances.

# Appendices
This folder contains two annexes mentioned in the paper:
- Appendix A: This document contains paraphrases with errors categorized by type of error. Each paraphrase is annotated with an error label.
- Appendix B: This document contains sample paraphrases labeled with different errors. This document illustrates the co-occurrence of several errors in the same paraphrase.

# TPME Dataset
A dataset of annotated paraphrases. The *Transformer-based Paraphrasing Model Errors (TPME)* dataset consists of 4790 annotated paraphrases, where the paraphrases have been labeled with a series of errors and explanations(sentence in simple natural English that justifies the error label). Each paraphrase is annotated with at least one of the following labels:
- Correct
- Semantic
- Spelling
- Grammmar
- Redundant
- Duplication
- Incoherent
- Punctuation
- Wrong Slot
- Slot Addition
- Slot omission
- Wordy
- Answering
- Questioning
- Homonym
- Acronym

## Possible use cases
- Automatic Paraphrasing errors Detection
- Paraphrase Generation
- Training data for Chatbots
- Entity recognition


## More information
You can contact me via audayberro (at) gmail.com

For more information please refer to our papar. Please also cite the following paper if you are using the dataset in your research:

```sh
@inproceedings{loremipsumdolor,
  title={A study of errors in Transformer-based Paraphrase Generation models},
  author={Berro, Auday and Benatallah, Boualem and Benabdeslem, Khalid},
  booktitle={lorem ipsum dolor},
  year={2023}
```

[snips]: <https://github.com/snipsco/snips-nlu>
