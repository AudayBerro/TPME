import pandas as pd
import json

"""
    Extract utterances from SNIPS json excerpt and save in csv format: utterance, list_of_slots.
    Extract the raw JSON data downloaded from the following link https://github.com/sonos/nlu-benchmark/tree/master/2017-06-custom-intent-engines
        { "data": [
                {"text": "Book a reservation for "},
                {"text": "seven", "entity": "party_size_number"},
                {"text": " people at a "},
                {"text": "bakery", "entity": "restaurant_type"},
                {"text": " in "},
                {"text": "Osage City", "entity": "city"},
           ]
        }
    
    The script parses 2 raw JSON datasets(BookRestaurant,GetWeather) and stores the respective datasets in a csv file with the following header format: utterance, list_of_slots
    
        e.g. "Book a reservation for  seven  people at a  bakery  in  Osage City","seven || party_size_number","bakery || restaurant_type","Osage City || city",
"""

def read_raw_data( file_name, json_key,intent):
    """
    Reads the raw SNIPS JSON dataset, extracts the statement with its respective required slots and converts the outcome into pandas dataframe. 
    :param file_name: path to the raw SNIPS dataset to be parsed extracted from https://github.com/sonos/nlu-benchmark/tree/master/2017-06-custom-intent-engines
    :param json_key: python string that specify the JSON object to read
    :param intent: intent of the current files, e.g. Bookrestaurant or GetWeather
    :return a pandas dataframe containing the parsed data with the following header: utterance, list_of_slots
    """

    utterances = dict()
    
    with open( file_name, 'r') as f:
        snips_dataset = json.load(f)

        # extract utterances from SNIPS train_BookRestaurant.json datset 
        for data in snips_dataset[ json_key ]:
            
            # extract the entity and its respective raw text chunk
            for data_field,text_field in data.items():
                entities = list()# utterance entities in order
                row = list()
                # if entity exist append
                for text in text_field:
                    row.append(text['text'])
                    # concatenate text_chunk with the entity, e.g: { "text": "seven", "entity": "party_size_number" }
                    #   => "seven || party_size_number"
                    if 'entity' in text:
                        entities.append(
                            text['text']+" || "+text['entity']
                        )
                
                row = listToStr = ' '.join(map(str, row))
                # entities.insert(0,utterance)
                utterances[row] = entities
    
    # df = pd.DataFrame({'utterance' : utterances.keys() , 'list_of_slots' : utterances.values() })
    df = pd.DataFrame.from_dict(utterances.items())
    df.columns = ['utterance', 'list_of_slots']
    return df

def extract_data():
    #parse GetWeather dataset https://github.com/sonos/nlu-benchmark/blob/master/2017-06-custom-intent-engines/GetWeather/train_GetWeather.json
    json_key = 'GetWeather'
    df = read_raw_data('./raw_data/weather.json',json_key)
    file_name = "./extracted_data/weather_dataset.csv"
    df.to_csv( file_name , sep=',', encoding='utf-8', index=False)
    
    #parse BookRestaurant dataset https://github.com/sonos/nlu-benchmark/blob/master/2017-06-custom-intent-engines/BookRestaurant/train_BookRestaurant.json
    json_key = 'BookRestaurant'
    df = read_raw_data('./raw_data/book_restaurant.json',json_key)
    file_name = "./extracted_data/book_restaurant.csv"
    df.to_csv( file_name , sep=',', encoding='utf-8', index=False)

if __name__ == '__main__':
    extract_data()
