import argparse
import pickle
import sys

from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer
from transformers import BertTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers import EvalPrediction

from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import torch

"""
    Fine-tuning BERT (and friends) for multi-label text classification.
    Source code from: https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/BERT/Fine_tuning_BERT_(and_friends)_for_multi_label_text_classification.ipynb#scrollTo=kLB3I4FKZ5Lr
"""

def clean_columns(df, desired_columns=['utterance', 'paraphrase', 'list_of_slots', 'error_category']):
    """
    Clean a pandas dataframe by removing undesired columns.

    :args
        - df (pandas.DataFrame): The pandas dataframe to be cleaned.
        - desired_columns (list): A list containing the desired columns to keep.

    :returns
        pandas.DataFrame: A cleaned dataframe with only the specified columns.
    """
    unwanted_columns = list(set(df.columns) - set(desired_columns))# get the difference between df.columns and the list of desired columns.
    
    df.drop(unwanted_columns, axis=1, inplace=True)# remove the unwanted columns from the dataframe.

    return df


def error_label_to_column(df):
    """
    Create 16 new columns and assign them a default value of 0 in a pandas dataframe. For each error label in the 'error_category' column,
    the value is replaced by 1 in the corresponding newly added column. These 16 added columns will serve as output label classes for model training.

    :args
        - df (pandas.DataFrame): The pandas dataframe to process.

    :returns
        pandas.DataFrame: A dataframe with the following header format:
            utterance, paraphrase, list_of_slots, error_category, semantic, spelling, grammar, redundant, duplication, incoherent, punctuation,
            wrong slot, slot addition, slot omission, wordy, answering, questioning, homonym, acronym, correct
    """
    new_columns = [
        'semantic', 'spelling', 'grammar', 'redundant', 'duplication', 'incoherent', 'punctuation', 'wrong slot',
        'slot addition', 'slot omission', 'wordy', 'answering', 'questioning', 'homonym', 'acronym', 'correct'
    ]

    for column in new_columns:
        df[column] = 0.0
    
    return df

def normalize_error_column(df):
    """
    Convert the values of the 'error_category' column to a list of errors, e.g., convert "semantic, wordy, slot addition" to ['semantic', 'wordy', 'slot addition'].

    :args
        - df (pandas.DataFrame): The pandas dataframe to process.

    :returns
        pandas.DataFrame: A dataframe with the 'error_category' column formatted as a list of errors.
    """

    labels_list = {
        'semantic', 'spelling', 'grammar', 'redundant', 'duplication', 'incoherent', 'punctuation', 'wrong slot',
        'slot addition', 'slot omission', 'wordy', 'answering', 'questioning', 'homonym', 'acronym', 'correct'
    }

    for i, row in df.iterrows():
        row_value = row['error_category']  # extract the current row value, e.g., semantic, wordy, slot addition
        row_value = row_value.lower().split(",")
        row_value = [i.strip() for i in row_value]  # remove leading and trailing whitespace

        # Check that the value of the 'error_category' column is not wrong and that it contains required labels.
        # It is possible that we have an empty cell or wrong data in the current row.
        if set(row_value).issubset(labels_list):
            for label in row_value:
                df.at[i, label] = 1.0

            df.at[i, 'error_category'] = f"{row_value}"
    return df


def id_to_label(dataset, cols_to_ignore):
    """
        This function extracts labels from a dataframe and creates two dictionaries that match integers to labels and vice versa.

        :args
            - dataset: a Huggingface dataset dictionary(datasets.dataset_dict.DatasetDict).
            - cols_to_remove: a python list of labels to ignore. E.g. if cols_to_ignore = ['utterance', 'paraphrase', 'list_of_slots'], these 3 cols wil be omitted in labels, id2label and label2id.

        :return
            Return three python lists:
                - labels: a python list of strings holding the labels
                - id2label: a python dictionary that maps integers to labels. The keys are the integers and the values are the labels.
                - label2id: a python dictionary that maps  labels to integers. The keys are the labels and the values are the integers.
    """
    
    # Create a list that contains the labels, as well as 2 dictionaries that map labels to integers and back.
    labels = [label for label in dataset['train'].features.keys() if label not in cols_to_ignore]
    id2label = {idx:label for idx, label in enumerate(labels)}
    label2id = {label:idx for idx, label in enumerate(labels)}
    return labels, id2label, label2id


def multi_label_metrics(predictions, labels, threshold=0.5):
    """
    Calculate evaluation metrics for multilabel classification.

    :args
        - predictions (torch.Tensor or array-like): Predicted probabilities from the model.
        - labels (array-like): Ground truth labels.
        - threshold (float, optional): Threshold for converting probabilities to binary predictions. Default is 0.5.

    :returns
        dict: A dictionary containing the following metrics:
            - 'f1': Micro-average F1 score.
            - 'roc_auc': Micro-average ROC AUC score.
            - 'accuracy': Accuracy score.

    :note
        This function assumes binary predictions and ground truth labels (0 or 1) and is suitable for multilabel classification problems.

    :source
        [Longformer Multilabel Classification](https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/)
    """
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    metrics = {'f1': f1_micro_average,
               'roc_auc': roc_auc,
               'accuracy': accuracy}
    return metrics

def compute_metrics(p: EvalPrediction):
    """
    Compute evaluation metrics for multilabel classification.

    :args
        p (EvalPrediction): An instance of EvalPrediction containing predictions and label_ids.

    :returns
        dict: A dictionary containing the following metrics:
            - 'f1': Micro-average F1 score.
            - 'roc_auc': Micro-average ROC AUC score.
            - 'accuracy': Accuracy score.

    :note
        This function assumes binary predictions and ground truth labels (0 or 1) and is suitable for multilabel classification problems.
    """
    preds = p.predictions[0] if isinstance(p.predictions, 
            tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds, 
        labels=p.label_ids)
    
    print(f"\n=== Result prediction:   ===\n{result}\n")
    return result


def process_data(sent1, sent2, tokenizer, max_length):
    """
        This function tokenize al sentneces in the given dataframe.

        :args
            - df: pandas dataframe.
            - tokenizer: a BERT tokenizer instance, e.g. bert-base-cased
            - max_length: maximum length of a sentence.
    """
    """
    Tokenize input sentences using a BERT tokenizer.

    :args
        - sent1 (str): The first sentence to be tokenized.
        - sent2 (str): The second sentence to be tokenized.
        - tokenizer: A BERT tokenizer instance, e.g., bert-base-cased.
        - max_length (int): Maximum length of a sentence.

    :returns
        dict: A dictionary containing the following tokenized representations:
            - 'input_ids': Input IDs of the tokenized sentences.
            - 'attention_mask': Attention mask indicating the presence of tokens.

    :Note
        This function uses the provided BERT tokenizer to encode the sentences with special tokens ([CLS] and [SEP]),
        pads the sequences to the specified maximum length, and returns the tokenized representations as PyTorch tensors.

    :Source
        - [BERT Tokenization](https://albertauyeung.github.io/2020/06/19/bert-tokenization.html/)
    """

    # Encode the sentence
    encoded = tokenizer(
        text=[sent1,sent2],  # the sentence to be encoded
        add_special_tokens=True,  # Add [CLS] and [SEP]
        max_length = 64,  # maximum length of a sentence
        pad_to_max_length=True,  # Add [PAD]s
        return_attention_mask = True,  # Generate the attention mask
        return_tensors = 'pt',  # ask the function to return PyTorch tensors
    )

    # Get the input IDs and attention mask in tensor format
    input_ids = encoded['input_ids']
    attn_mask = encoded['attention_mask']

def split_and_convert_data(df):
    """
    Split a pandas dataframe into training and test sets (80%, 20%) and convert them into a Huggingface dataset.

    :args
        - df (pandas.DataFrame): The pandas dataframe to split and convert.

    :returns
        datasets.dataset_dict.DatasetDict: A Huggingface dataset dictionary with the following layout:
            DatasetDict({
                'train': Dataset({ 'features': [...], 'num_rows': ... }),
                'validation': Dataset({ 'features': [...], 'num_rows': ... })
            })

    :note
        This function uses the `train_test_split` method from scikit-learn to divide the dataframe into training and test sets.
        The resulting datasets are then converted into Huggingface datasets using the `from_pandas` method.
    """
    
    # divide the dataframe into training and test sets with a size of 80% and 20% respectively (test_size=0.20)
    train, test = train_test_split(df, test_size=0.20, random_state=42)

    # convert into Huggignface datasets: https://discuss.huggingface.co/t/from-pandas-dataframe-to-huggingface-dataset/9322
    tds = Dataset.from_pandas(train)
    vds = Dataset.from_pandas(test)

    ds = DatasetDict()

    ds['train'] = tds
    ds['validation'] = vds

    return ds

def tokenize_data(dataset, tokenizer, labels):
    """
    Tokenize a Huggingface dataset dictionary using a BERT tokenizer and include labels in the encoding.

    :args
        - dataset (datasets.dataset_dict.DatasetDict): A Huggingface dataset dictionary.
        - tokenizer: An instance of a BERT tokenizer, e.g., tokenizer = BertTokenizer.from_pretrained("bert-base-uncased").
        - labels (list): A Python list containing the required labels for the batch_labels. This list contains the error taxonomy.

    :returns
        transformers.tokenization_utils_base.BatchEncoding: A BatchEncoding object containing encoded sentences and labels.
    """

    def preprocess_data(examples):

        # Take a batch of texts

        # Extract utternace and paraphrase  
        utterance = examples['utterance']
        paraphrase = examples['paraphrase']

        # Pair up the 2 lists, necessary to have the following schema later: [CLS] utterance [SEP] paraphrase [SEP]
        # The below variable text is a list of a list of pair of utterance and paraprhase e.g. [ [U1,P1], [U2,P2],..., [Un, Pn]]
        text = [ [u,p] for u,p in zip(utterance,paraphrase)]

        # encode them
        encoding = tokenizer(text, padding="max_length", truncation=True, max_length=512)

        # add labels
        labels_batch = {k: examples[k] for k in examples.keys() if k in labels}

        # create numpy array of shape (batch_size, num_labels)
        labels_matrix = np.zeros((len(text), len(labels)))

        # fill numpy array
        for idx, label in enumerate(labels):
            labels_matrix[:, idx] = labels_batch[label]

        encoding["labels"] = labels_matrix.tolist()#vector of labels value e.g. [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        
        return encoding
    
    encoded_dataset = dataset.map(preprocess_data, batched=True, remove_columns= dataset['train'].column_names)
    return encoded_dataset

def write_history(train_history, pickle_path):
    """
        This function write a transformers.trainer.state.log_history into a file.
        
        :args
            - train_history: a trainer.state.log_history object
            - pickle_path: the name of the pickle object 
    """
    
    with open(pickle_path, 'wb') as fp:
        pickle.dump(train_history, fp)
        print('Done writing training history into a binary file')

def main(df):
    """
        A wrapper function to fine-tune BERT on the multi-labeling task by adding a linear layer on top of BERT which is used to produce a tensor of shape (batch_size, num_labels).

        :args
            - df: a pandas dataframe containing the dataset that will be used to fine-tune BERT.

        :return
            A pandas dataframe.
    """

    print(f"Some samples:\n {df.head(5)}")

    #convert the pandas dataframe df to a Huggingface dataset
    dataset = split_and_convert_data(df)

    # # Create a list that contains the errors labels only, as well as 2 dictionaries that map labels to integers and back.
    # print(f"Extract the error labels and build two dictionaries that transform the labels into integers and vice versa:")
    # labels = [label for label in df.columns.tolist() if label not in ['utterance', 'paraphrase', 'list_of_slots', 'error_category']]
    # id2label = {idx:label for idx, label in enumerate(labels)}
    # label2id = {label:idx for idx, label in enumerate(labels)}
    
    cols_2_ignore = ['utterance', 'paraphrase', 'list_of_slots', 'error_category','__index_level_0__']
    labels, id2label, label2id = id_to_label(dataset, cols_2_ignore)

    print(f"...extracted labels: {labels}")
    print(f"...id2label: {id2label}")
    print(f"...label2id: {label2id}")

    # Preprocess data: As models like BERT don't expect text as direct input, but rather input_ids, etc., we tokenize the text using the tokenizer.
    # Here we use the AutoTokenizer which will automatically load the appropriate tokenizer based on the choosen checkpoint on the hub.
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    encoded_dataset = tokenize_data(dataset, tokenizer,labels)

    # check a sample
    example = encoded_dataset['train'][0]
    print(f"Encoded sample: {example.keys()}")
    print(f"Decode the sample: {tokenizer.decode(example['input_ids'])}")
    print(f"Sample encoded label: {example['labels']}")

    # check the sample labeling
    decoded_label = [id2label[idx] for idx, label in enumerate(example['labels']) if label == 1.0]
    print(f"Sample decoded label: {decoded_label}")

    encoded_dataset.set_format("torch")

    # Define model: define a model that includes a pre-trained base (i.e. the weights from bert-base-uncased) are loaded, with a random initialized classification head (linear layer) on top.
    # One should fine-tune this head, together with the pre-trained base on a labeled dataset.
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", 
        problem_type="multi_label_classification", 
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id
    )

    # Train the model
    batch_size = 64
    metric_name = "f1"
    num_train_epochs = 10

    args = TrainingArguments(
        f"bert-finetuned-multi-label",
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name
    )

    print(f"encoded_dataset.type: {encoded_dataset['train'][0]['labels'].type()}")
    print(f"Encoded data sample: {encoded_dataset['train']['input_ids'][0]}")

    # Forward pass
    outputs = model(input_ids=encoded_dataset['train']['input_ids'][0].unsqueeze(0), labels=encoded_dataset['train'][0]['labels'].unsqueeze(0))
    print(outputs)

    # Start the training
    trainer = Trainer(
        model,
        args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()

    trainer. save_model("./finetuned_model/")
    
    train_history = trainer.state.log_history
    print(f"HISTORY:\n{train_history}")
    
    history_path = f"./train_history/BERT_e-{num_train_epochs}_b-{batch_size}"#e for epoch and b for batch
    write_history(train_history, history_path)

    return trainer

def load_data(file_name):
    """
        A wrapper function to load the csv file to process.

        :args
            - file_name: Path to where the dataset that will be used to fine-tune BERT is stored. This must be a csv file.

        :return
            A pandas dataframe.
    """

    __labels = [
        'semantic', 'spelling', 'grammar', 'redundant', 'duplication', 'incoherent', 'punctuation', 'wrong slot', 'slot addition', 'slot omission', 'wordy', 'answering', 'questioning', 'homonym', 'acronym', 'correct'
    ]

    #read data
    df = pd.read_csv(file_name, sep = ',',na_filter= False)

    #print columns
    # print(df.columns)
    
    #remove unwanted columns
    df = clean_columns(df)

    #add the 16 output classes
    df = error_label_to_column(df)

    df = normalize_error_column(df)

    return df

def get_prediction(utterance, paraphrase, model, tokenizer):
    """
        This function annotates a paraphrase with a label predicted using the refined BERT model.

        :args
            - utterance: the original statement from which the paraphrase was generated.
            - paraphrase: the paraphrase to be labelled.
            - model: an instance of the fine-tuned BERT model on multi-labeling task.
            - tokenizer: an instnace of a BERT tokenizer.
        
        :return
            Two pairs of python lists. The first list contains the predicted labels and the second list contains the logits.
    """

    encoding = tokenizer(utterance, paraphrase,return_tensors="pt")
    encoding = {k: v.to(model.device) for k,v in encoding.items()}

    outputs = model(**encoding)
    
    #The logits that come out of the model are of shape (batch_size, num_labels).
    #As we are only forwarding a single sentence through the model, the batch_size equals 1.
    #The logits is a tensor that contains the (unnormalized) scores for every individual label.
    logits = outputs.logits
    print(f"logits.shape: {logits.shape}")

    # apply sigmoid + threshold
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(logits.squeeze().cpu())
    predictions = np.zeros(probs.shape)
    print(f"predictions = {predictions}")
    predictions[np.where(probs >= 0.5)] = 1
    print(f"predictions[1] = {predictions}")
    # turn predicted id's into actual label names
    predicted_labels = [id2label[idx] for idx, label in enumerate(predictions) if label == 1.0]
    print(f"predicted_labels: {predicted_labels}")

if __name__ == "__main__":
    ### To exectur this script first activate the virtual env: source ../virtualenvironment/myenvWithPython3/bin/activate

    parser = argparse.ArgumentParser()

    parser.add_argument('-f', '--filename', help = 'The name of the csv file to process.', required = True)

    args = parser.parse_args()

    #read data
    df = load_data(args.filename)

    ################ TRAINING CODE ################
    trainer = main(df)

    ################ INFERENCE CODE ################
    #model = AutoModelForSequenceClassification.from_pretrained("./finetuned_model/")
    #tokenizer = BertTokenizer.from_pretrained("./finetuned_model/")
    
    #ataset = split_and_convert_data(df)
    #cols_2_ignore = ['utterance', 'paraphrase', 'list_of_slots', 'error_category','__index_level_0__']
    #_, id2label,_ = id_to_label(dataset, cols_2_ignore)
    #print(id2label)

    #test the model on a new sentence:
    #utterance = "Tell me the weather forecast for  Roseburg ,  Iowa"
    #paraphrase = "Tell me the weather forecast for  Roseburg ,  Iowa...~??"

    #get_prediction(utterance, paraphrase, model, tokenizer)