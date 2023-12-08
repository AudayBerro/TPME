import pandas as pd

"""
	Error Analysis Script

	This script performs error analysis on datasets, specifically focusing on two key tasks:

	1. Counting Incoherent and Redundant Errors:
	   The function count_error_instances(df) takes a DataFrame df as input and assesses the count of instances
	   characterized as 'Incoherent' and 'Redundant' errors. It returns a tuple containing the counts for each error type.

	2. Counting Specific Errors in a CSV File:
	   The function count_specific_errors(file_name, errors_to_find) reads a CSV file specified by file_name,
	   where the data is expected to be in a single column separated by tabs. It counts the occurrences of specific error
	   labels provided in the errors_to_find list. The result is returned as a dictionary with error labels as keys and
	   corresponding counts as values.

	Note:
	   Ensure that the CSV files follow the expected formats for accurate error analysis.

	Author: Auday Berro
	Date: 05/12/2023
"""

def count_error_instances(df):
    # Check for the number of Incoherent and Redundant errors instances
    incoherent = df[df['incoherent'] == 1].shape[0]
    redundant = df[df['redundant'] == 1].shape[0]
    
    return incoherent, redundant

def count_specific_errors(file_name, errors_to_find):
    """
    Count the number of rows in a DataFrame with specific error labels.

    :args
    - file_name (str): The path to the CSV file.
    - errors_to_find (list): A list of error labels to count.

    :returns
    - int: The total number of rows with the specified error labels.
    """
    # To avoid tokenization errors in the df with a single column accepting multi-labels,
    # set the separator to '\t', header to None, and rename the column as 'labels'.
    df = pd.read_csv(file_name, sep='\t', header=None, names=['labels'])

    # Count rows with specific error labels
    total_matching_count = {err: df['labels'].eq(err).sum() for err in errors_to_find}

    return total_matching_count

if __name__ == "__main__":
	try:
		# Try reading the file with pandas
		file_name = "upset_data.csv"
		df = pd.read_csv(file_name)
		
		a,b = count_error_instances(df)
		print(f"Incoherent: {a}")
		print(f"Redundant: {b}")
		
		file_name = 'TPME_labels_only.csv'
		errors_to_find = ['semantic', 'incoherent', 'spelling']
		result = count_specific_errors(file_name,errors_to_find)
		print(result)
		
	except pd.errors.ParserError as e:
		# Print the error message
		print("Error:", e)

		# Investigate the problematic line (line 21 in this case)
		with open(file_name, 'r') as file:
		    lines = file.readlines()
		    problematic_line = lines[20]  # Assuming line numbering starts from 1
		    print("Problematic Line:", problematic_line)
