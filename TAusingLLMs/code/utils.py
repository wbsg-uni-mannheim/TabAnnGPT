import numpy as np
import pickle
import re
import pandas as pd
import json
from langchain.embeddings.openai import OpenAIEmbeddings
import os
from dotenv import dotenv_values

config = dotenv_values("/full/path/to/file/key.env")
os.environ['OPENAI_API_KEY'] = config["OPENAI_API_KEY"]
OPENAI_API_KEY = config["OPENAI_API_KEY"]

def save_pickle_file(file_name, output):
    # Save table predictions in a file:
    f = open(file_name,'wb')
    pickle.dump(output,f)
    f.close()

def load_pickle_file(file_name):
    # Load .pkl file
    with open(file_name, "rb") as f:
        file = pickle.load(f)
    return file

def textada_embeddings(text, OPENAI_API_KEY):
    # Embed some text with text-ada
    model_name = 'text-embedding-ada-002'
    embed = OpenAIEmbeddings(model=model_name, openai_api_key=OPENAI_API_KEY)

    return embed.embed_documents(text)

# Load data functions
def load_cpa_dataset_column(dataset, headers):
    # Import labels to text
    f = open(f'../data/labels_to_text_{dataset}-cpa.txt')
    labels_to_text = json.load(f)

    with open(f'../data/{dataset}-cpa-train{headers}-column.pkl', "rb") as f:
        train = pickle.load(f)
    with open(f'../data/{dataset}-cpa-test{headers}-column.pkl', "rb") as f:
        test = pickle.load(f)

    examples = [example[2] for example in test ]
    labels = [example[3] for example in test ]

    train_examples = [ example[2] for example in train ]
    train_labels = [ labels_to_text[example[3]] for example in train ]
    
    text_to_label = {labels_to_text[label]: label for label in labels_to_text}
    labels_joined = ", ".join([labels_to_text[l] for l in labels_to_text])

    return examples, labels, train_examples, train_labels, labels_to_text, text_to_label, labels_joined, train, test

def load_cta_dataset_column(dataset, headers):
    if dataset != "sportstables":
        f = open(f'../data/labels_to_text_{dataset}-cta.txt')
        labels_to_text = json.load(f)
    else:
        f = open(f"../data/sportstables-labels/sportstables_all_labels.txt", 'r')
        all_labels = [line.split('\n')[0] for line in f.readlines()]
        labels_to_text = {label: label for label in all_labels}
    
    with open(f'../data/{dataset}-cta-train{headers}-column.pkl', "rb") as f:
        train = pickle.load(f)
    with open(f'../data/{dataset}-cta-test{headers}-column.pkl', "rb") as f:
        test = pickle.load(f)
    
    examples = [example[2] for example in test ]
    labels = [example[3] for example in test ]

    train_examples = [ example[2] for example in train ]
    train_labels = [ labels_to_text[example[3]] for example in train ]

    text_to_label = {labels_to_text[label]: label for label in labels_to_text}
    labels_joined = ", ".join([labels_to_text[l] for l in labels_to_text])

    return examples, labels, train_examples, train_labels, labels_to_text, text_to_label, labels_joined, train, test

def load_cta_dataset(dataset, headers):
    # Import labels to text
    if dataset != "sportstables":
        f = open(f'../data/labels_to_text_{dataset}-cta.txt')
        labels_to_text = json.load(f)
        # all_labels = [labels_to_text[l] for l in labels_to_text]
    else:
        f = open(f"../data/sportstables-labels/sportstables_all_labels.txt", 'r')
        all_labels = [line.split('\n')[0] for line in f.readlines()]
        labels_to_text = {label: label for label in all_labels}
             

    with open(f'../data/{dataset}-cta-train{headers}.pkl', "rb") as f:
        train = pickle.load(f)
    with open(f'../data/{dataset}-cta-test{headers}.pkl', "rb") as f:
        test = pickle.load(f)

    text_to_label = {labels_to_text[label]: label for label in labels_to_text}
    labels_joined = ", ".join([labels_to_text[l] for l in labels_to_text])

    examples = [example[1] for example in test ]
    labels = [l for example in test for l in example[2]]
    test_table_type_labels = [ example[3] for example in test ]

    train_examples = [ example[1] for example in train ]
    train_example_labels = []
    for table in train:
        col_labels = """"""
        for i, l in enumerate(table[2]):
            col_labels += f"""Column {i+1}: {labels_to_text[l]}\n"""
        train_example_labels.append(col_labels.strip())
    train_table_type_labels = [ example[3] for example in train ]

    return examples, labels, test_table_type_labels, train_examples, train_example_labels, train_table_type_labels, labels_to_text, text_to_label, labels_joined, train, test

def load_cpa_dataset(dataset, headers, cta):
    # Import labels to text
    f = open(f'../data/labels_to_text_{dataset}-cpa.txt')
    labels_to_text = json.load(f)

    with open(f'../data/{dataset}-cpa-train{headers}.pkl', "rb") as f:
        train = pickle.load(f)
    with open(f'../data/{dataset}-cpa-test{headers}.pkl', "rb") as f:
        test = pickle.load(f)

    text_to_label = {labels_to_text[label]: label for label in labels_to_text}
    labels_joined = ", ".join([labels_to_text[l] for l in labels_to_text])

    examples = [example[1] for example in test ]
    labels = [l for example in test for l in example[2] if l != ""]
    test_table_type_labels = [ example[3] for example in test ]

    train_examples = [ example[1] for example in train ]
    train_example_labels = []

    for table in train:
        col_labels = """"""
        if cta:
            col_labels += f"""Column 1: name\n"""
        for i, l in enumerate(table[2]):
            if i != 0:
                col_labels += f"""Column {i+1}: {labels_to_text[l]}\n"""
        train_example_labels.append(col_labels.strip())
    train_table_type_labels = [ example[3] for example in train ]
    
    return examples, labels, test_table_type_labels, train_examples, train_example_labels, train_table_type_labels, labels_to_text, text_to_label, labels_joined, train, test

# Evaluation functions
def decimal(num):
    return np.around(num*100, decimals=2, out=None)

def calculate_f1_scores(y_tests, y_preds, num_classes, types):

    y_tests = [types.index(y) for y in y_tests]
    y_preds = [types.index(y) for y in y_preds]
    
    #Confusion matrix
    cm = np.zeros(shape=(num_classes,num_classes))
    
    for i in range(len(y_tests)):
        cm[y_preds[i]][y_tests[i]] += 1
        
    report = {}
    
    for j in range(len(cm[0])):
        report[j] = {}
        report[j]['FN'] = 0
        report[j]['FP'] = 0
        report[j]['TP'] = cm[j][j]

        for i in range(len(cm)):
            if i != j:
                report[j]['FN'] += cm[i][j]
        for k in range(len(cm[0])):
            if k != j:
                report[j]['FP'] += cm[j][k]

        precision = report[j]['TP'] / (report[j]['TP'] + report[j]['FP'])
        recall = report[j]['TP'] / (report[j]['TP'] + report[j]['FN'])
        f1 = 2*precision*recall / (precision + recall)
        
        if np.isnan(f1):
            f1 = 0
        if np.isnan(precision):
            f1 = 0
        if np.isnan(recall):
            f1 = 0

        report[j]['p'] =  precision
        report[j]['r'] =  recall
        report[j]['f1'] = f1
    
    all_fn = 0
    all_tp = 0
    all_fp = 0

    for r in report:
        if r != num_classes-1:
            all_fn += report[r]['FN']
            all_tp += report[r]['TP']
            all_fp += report[r]['FP']
        
    class_f1s = [ report[class_]['f1'] for class_ in report]
    class_p = [ 0 if np.isnan(report[class_]['p']) else report[class_]['p'] for class_ in report]
    class_r = [ 0 if np.isnan(report[class_]['r']) else report[class_]['r'] for class_ in report]
    macro_f1 = sum(class_f1s[:-1]) / (num_classes-1)
    
    p =  sum(class_p[:-1]) / (num_classes-1)
    r =  sum(class_r[:-1]) / (num_classes-1)
    micro_f1 = all_tp / ( all_tp + (1/2 * (all_fp + all_fn) )) 
    
    per_class_eval = {}
    for index, t in enumerate(types[:-1]):
        per_class_eval[t] = {"Precision":class_p[index], "Recall": class_r[index], "F1": class_f1s[index]}
    
    evaluation = {
        "Micro-F1": micro_f1,
        "Macro-F1": macro_f1,
        "Precision": p,
        "Recall": r
    }
    
    return [ evaluation, per_class_eval]


def map_cpa_to_labels(preds, test, text_to_label):
    # Map predictions to label space
    predictions = []
    i=0
    num = 0
    for j, table_preds in enumerate(preds):
        # How many columns does the table have? : To control cases when less/more classes are returned
        table_number = len(test[j][2])
        
        if "Class:" in table_preds:
            table_preds = table_preds.split("Class:")[1]
        
        #Break predictions into either \n or ,
        if ":" in table_preds or "-" in table_preds:
            if ":" in table_preds:
                separator = ":"
                start = 1
                end = table_number+1
            else:
                separator = "-"  
                start = 1
                end = table_number+1
        else:
            separator = ","
            start = 0
            end = table_number
            
        col_preds = table_preds.split(separator)[start:end]
        
        for pred in col_preds[:table_number-1]:
            i+=1
            
            # Remove break lines
            if "\n" in pred:
                pred = pred.split('\n')[0].strip()
            # Remove commas
            if "," in pred:
                pred = pred.split(",")[0].strip()
            # Remove paranthesis
            if '(' in pred:
                pred = pred.split("(")[0].strip()
            #Remove points
            if '.' in pred:
                pred = pred.split(".")[0].strip()

            #Remove punctuation
            pred = re.sub(r'[^\w\s]','',pred)
            # Lower-case prediction
            pred = pred.strip().lower()
            
            if pred in text_to_label:
                predictions.append(text_to_label[pred])
            else:
                fin = ""
                for la in text_to_label:
                    if la in pred:
                        fin = la
                        break
                if fin == "":
                    fin2 = ""
                    for la in text_to_label:
                        if pred in la:
                            fin2 = la
                            break
                    
                    if fin2=="":
                        # print(f"For test example {i} out of label space prediction: {pred}")
                        predictions.append('-')
                        num +=1
                    else:
                        predictions.append(text_to_label[fin2])
                else:
                    predictions.append(text_to_label[fin])
        
        # If more/less predictions for table
        if len(col_preds) < table_number-1:
            for m in range(0, table_number-1-len(col_preds)):
                predictions.append('-')
                num +=1
                i+=1
    return predictions, num

def map_cta_labels(preds, test, text_to_label):
    # Map predictions to label space
    predictions = []
    i=0
    num = 0
    for j, table_preds in enumerate(preds):
        # How many columns does the table have? : To control cases when less/more classes are returned
        table_number = len(test[j][2])
        
        if "Class:" in table_preds:
            table_preds = table_preds.split("Class:")[1]
        
        #Break predictions into either \n or ,
        if ":" in table_preds or "-" in table_preds:
            if ":" in table_preds:
                separator = ":"
                start = 1
                end = table_number+1
            else:
                separator = "-"  
                start = 1
                end = table_number+1
        else:
            separator = ","
            start = 0
            end = table_number
            
        col_preds = table_preds.split(separator)[start:end]
        
        for pred in col_preds[:table_number]:
            i+=1
            
            # Remove break lines
            if "\n" in pred:
                pred = pred.split('\n')[0].strip()
            # Remove commas
            if "," in pred:
                pred = pred.split(",")[0].strip()
            # Remove paranthesis
            if '(' in pred:
                pred = pred.split("(")[0].strip()
            #Remove points
            if '.' in pred:
                pred = pred.split(".")[0].strip()

            #Remove punctuation
            pred = re.sub(r'[^\w\s]','',pred)
            # Lower-case prediction
            pred = pred.strip().lower()
            
            if pred in text_to_label:
                predictions.append(text_to_label[pred])
            else:
                fin = ""
                for la in text_to_label:
                    if la in pred:
                        fin = la
                        break
                if fin == "":
                    fin2 = ""
                    for la in text_to_label:
                        if pred in la:
                            fin2 = la
                            break
                    
                    if fin2=="":
                        # print(f"For test example {i} out of label space prediction: {pred}")
                        predictions.append('-')
                        num +=1
                    else:
                        predictions.append(text_to_label[fin2])
                else:
                    predictions.append(text_to_label[fin])
        
        # If more/less predictions for table
        if len(col_preds) < table_number:
            for m in range(0, table_number-len(col_preds)):
                predictions.append('-')
                num +=1
                i+=1
    return predictions, num

def clean_text(text):
        
    if(isinstance(text, dict)):
        text = ' '.join([ clean_text(v) for k, v in text.items()] )
    elif(isinstance(text, list)):
        text = map(clean_text, text)
        text = ' '.join(text)
        
    if pd.isnull(text):
        return ''
    
    #Remove excess whitespaces
    text = re.sub(' +', ' ', str(text)).strip()
    
    return text

def map_answers_column(preds, test, text_to_label, labels_to_text):
    predictions = []
    num = 0
    
    for i, pred in enumerate(preds):
        # Remove break lines
        if "\n" in pred:
            pred = pred.split('\n')[0].strip()
        # Remove commas
        if "," in pred:
            pred = pred.split(",")[0].strip()
        # Remove paranthesis
        if '(' in pred:
            pred = pred.split("(")[0].strip()
        #Remove points
        # if '.' in pred:
        #     pred = pred.split(".")[0].strip()

        #Remove punctuation
        # pred = re.sub(r'[^\w\s]','',pred)
        # Lower-case prediction
        pred = pred.strip().lower()
        
        labels_lowered = {lab.lower(): lab for lab in labels_to_text}

        if pred in text_to_label:
            predictions.append(text_to_label[pred])
        elif pred in labels_to_text:
            predictions.append(pred)
        elif pred in labels_lowered:
            predictions.append(labels_lowered[pred])
        else:
            fin = ""
            for la in text_to_label:
                if la in pred:
                    fin = la
                    break
            if fin == "":
                fin2 = ""
                for la in text_to_label:
                    if pred in la:
                        fin2 = la
                        break

                if fin2=="":
                    # print(f"For test example {i} out of label space prediction: {pred}")
                    predictions.append('-')
                    num +=1
                else:
                    predictions.append(text_to_label[fin2])
            else:
                predictions.append(text_to_label[fin])
                    
    return predictions, num


def map_sportstables(preds, all_labels, test):
    # Map predictions to label space
    predictions = []
    i=0
    num = 0
    count = 0
    for j, table_preds in enumerate(preds):
        # How many columns does the table have? : To control cases when less/more classes are returned
        table_number = len(test[j][2])

        if "Class:" in table_preds:
            table_preds = table_preds.split("Class:")[1]

        #Break predictions into either \n or ,
        if ":" in table_preds or "-" in table_preds:
            if ":" in table_preds:
                separator = ":"
                start = 1
                end = table_number+1
            else:
                separator = "-"  
                start = 1
                end = table_number+1
        else:
            separator = ","
            start = 0
            end = table_number

        col_preds = table_preds.split(separator)[start:end]

        # If main column twice
        col_preds_start = 0
        if "" in test[j][2]:
            col_preds_start = 1
            count += 1

        for pred in col_preds[col_preds_start:table_number]:
            i+=1

            pred = pred.strip()

            # Remove break lines
            if "\n" in pred:
                pred = pred.split('\n')[0].strip()
            # Remove commas
            if "," in pred:
                pred = pred.split(",")[0].strip()
            # Remove paranthesis
            if '(' in pred:
                pred = pred.split("(")[0].strip()

            # Lower-case prediction
            pred = pred.strip().lower()

            if pred in all_labels:
                predictions.append(pred)
            else:
                fin = ""
                for la in all_labels:
                    if la in pred:
                        fin = la
                        break
                if fin == "":
                    if pred == "baseball.team.caught_stealing":
                        predictions.append("baseball.team.cauught_stealing")
                    elif pred == "basketball.coach.franchise_seasons":
                        predictions.append("basketball.coach.franchise_sesaons")
                    elif pred == "hockey.player.penalties_minutes":
                        predictions.append("hockey.player.penaltie_minutes")
                    elif pred == "hockey.player.shot_on_goals":
                        predictions.append("hockey.player.short_on_goals")
                    elif pred == "basketball.team.free_throw_attempts_per_game":
                        predictions.append("basketball.team.three_throw_attempts_per_game")
                    elif pred == "soccer.player.name":
                        predictions.append("soocer.player.name")
                    else:
                        # print(f"For test example {i} out of label space prediction: {pred}")
                        predictions.append('-')
                        num +=1
                else:
                    predictions.append(fin)


        # If more/less predictions for table
        if len(col_preds)-col_preds_start < table_number-col_preds_start:
            for m in range(0, table_number-len(col_preds)):
                predictions.append('-')
                i+=1
        
    return predictions, num
