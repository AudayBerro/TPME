Utterance extraction from the [SNIPS][snips] dataset.  This folder contains the code we used to parse the raw JSON SNIPS excerpts retrieved from [SNIPS][snips] to extract the seed utterances used in our paper to generate the paraphrases(described in section 2.2 of the paper).


## raw_data
This folder contains raw JSON files retrieved from the  [SNIPS][snips] github repository.
| Dataset | link | Description |
| ------ | ------ | ------ |
| book_restaurant.json | [BookRestaurant][book-rest-url] | Crowdsourced dataset for the **BookRestaurant** intent|
| weather.json | [GetWeather][weather-url] | Crowdsourced dataset for the **GetWeather** intent|

## extracted_data
This folder contains the parsed raw data stored in **raw_data** folder using the  `parse_snips_dataset.py` script. The script reads each raw JSON file stored in the **raw_data** folder, extracts the utterances and their respective list of required slots, then saves the result in a csv file in the following format (*utterance*, *list_of_slots*).
The folder contain 2 datasets:
- **book_restaurant.csv** dataset extracted from book_restaurant.json
- **weather_dataset.csv** dataset extracted from weather.json


[snips]: <https://github.com/sonos/nlu-benchmark/tree/master/2017-06-custom-intent-engines>
[book-rest-url]: <https://github.com/sonos/nlu-benchmark/blob/master/2017-06-custom-intent-engines/BookRestaurant/train_BookRestaurant.json>
[weather-url]: <https://github.com/sonos/nlu-benchmark/blob/master/2017-06-custom-intent-engines/GetWeather/train_GetWeather.json>