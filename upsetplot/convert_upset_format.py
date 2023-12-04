import pandas as pd
import numpy as np
import argparse
import sys

""" 
    This script process raw text file and convert to UpSet Plot compatible input format. 
    UpSet Plot official website: https://upset.app/
    Draw the UpSet plot using the Intervene plateform: https://asntech.shinyapps.io/intervene/
    Each row in the raw text file contain only labels.
    e.g.
        semantic
        correct
        correct
        semantic, wordy, slot addition
        duplication
        slot omission, wrong slot

    Will convert the file to Binary type data format as stated in  https://asntech.shinyapps.io/intervene/ -> UpSet -> usage instructions -> Binary type data.
    An example file can be downloaded using the following link: https://asntech.shinyapps.io/intervene/_w_1e624542/mutations_glioblastoma_TCGA.csv

   In the binary input file each column represents a set, and each row represents an element. If a names is in the set then it is represented as a 1, else it is represented as a 0. 
   The output of the script is a csv file with the following header:
      semantic, spelling, grammar, redundant, duplication, homonym, answering, incoherent, wordy, acronym, slot omission, wrong slot, slot addition, punctuation, questioning


    run the script:  python3 convert_upset_format.py -r TPME_labels_only.txt
"""

def normalize_data(filename):
    """
        This function convert raw txt file to UpSet binary data format.
        :param filename: path to the file containing the raw data. A file with txt extension.
        return a pandas dataframe with the following header:
            correct, semantic, spelling, grammar, redundant, duplication, homonym, answering, incoherent, wordy, acronym, slot omission, wrong slot, slot addition, punctuation, questioning
    """

    tmp = {
        "correct": 0,
        "semantic": 0,
        "spelling": 0,
        "grammar": 0, 
        "redundant": 0,
        "duplication": 0, 
        "homonym": 0, 
        "answering": 0,
        "incoherent": 0,
        "wordy": 0,
        "acronym": 0,
        "slotomission": 0,
        "wrongslot": 0,
        "slotaddition": 0,
        "punctuation": 0,
        "questioning": 0
    }

    data = list()
    
    with open( filename , 'r') as raw_data:

        #process each raw and if a label is in the row then it is represented as a 1, else it is represented as a 0.
        for row in raw_data:
            labels = row.replace(" ", "").replace("\n", "").split(",")#get labels
            unique_l.update(labels)
            binary_row = tmp.copy()

            if "correct" in labels:
                binary_row["correct"] = 1
            else:
                for l in labels:
                    binary_row[l] = 1# If a label is in the row then it is represented as a 1, else it is represented as a 0.
            
            tmp_list = list( binary_row.values())#get row representation as a list of 0 and 1
            data.append(tmp_list)
    
    columns= [
        "correct",
        "semantic", 
        "spelling", 
        "grammar", 
        "redundant", 
        "duplication", 
        "homonym", 
        "answering", 
        "incoherent", 
        "wordy", 
        "acronym", 
        "slot omission", 
        "wrong slot", 
        "slot addition", 
        "punctuation", 
        "questioning"]
    
    
    df = pd.DataFrame( data, columns = columns, index = None)
    df.to_csv( f'./upset_data.csv' , sep=',', encoding='utf-8', index=False)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='raw text file to convert in UpSet-Plot compatible input format')
    parser.add_argument('-r', '--raw', help='text file containing the raw data: labels separated by comma.', required=True)

    args = parser.parse_args()

    normalize_data(args.raw)
