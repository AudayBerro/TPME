This folder contains the paraphrases genrated with the 4 selected transformer-based paraprhase generation models described in section 2 of the paper.


## Generated Paraphrases
This folder contains raw JSON files retrieved from the  [SNIPS][snips] github repository. Each data set is a csv file with the following format: (*utterance*, *paraphrase*, *list_of_slots*, *intent*)
- *utterance* is the seed sentence to be paraphrased.
- *paraphrase* this column will contain a generated paraphrase.
- *list_of_slots* contains any slot occuring in the utterance with its associated value.
- *intent* indicates the intention of the utterance, e.g. BookRestaurant or GetWeather.

| Datasets | Seed utterances Intent | Models| Number of paraphrases |
| ------ | ------ | ------ | ------ |
| PRISM_BR | *BookRestaurant* | Paraphrases generated with the PRISM model | 1500 |
| PRISM_GW | *GetWeather* | Paraphrases generated with the PRISM model | 1486 |
| NL_BR | *BookRestaurant* | Paraphrases generated with the NL-Augmenter model | 3000 |
| NL_GW | *GetWeather* | Paraphrases generated with the NL-Augmenter model | 2970 |
| T5_BR | *BookRestaurant* | Paraphrases generated with the fine-tuned T5 model | 3000 |
| T5_GW | *GetWeather* | Paraphrases generated with the fine-tuned T5 model | 2950 |
| PROT_BR | *BookRestaurant* | Paraphrases generated with the fine-tuned T5 model | 3000 |
| PROT_GW | *GetWeather* | Paraphrases generated with the fine-tuned T5 model | 2970 |
| GPT_BR | *BookRestaurant* | Paraphrases generated with GPT-turbo 3.5 | 1003 |
| GPT_GW | *GetWeather* | Paraphrases generated with GPT-turbo 3.5 | 1005 |
