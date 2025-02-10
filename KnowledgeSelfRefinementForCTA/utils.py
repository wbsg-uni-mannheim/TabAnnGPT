import pickle
import re
import pandas as pd
import json
from langchain.embeddings.openai import OpenAIEmbeddings
import sienna
import os
from dotenv import dotenv_values
import pdb
import json
from sklearn.metrics.pairwise import cosine_similarity

config = dotenv_values("/full/path/to/file/key.env")
os.environ['OPENAI_API_KEY'] = config["OPENAI_API_KEY"]
OPENAI_API_KEY = config["OPENAI_API_KEY"]

def parse_json(json_string):
    # Parse JSON string
    try:
        pred_j = json.loads(json_string)
    except Exception:
        try:
            pred_j = json.loads(("{"+json_string.split("{")[1].split("}")[0].strip()+"}").replace(",}", "}").replace("} \n{",", ").replace("}; {",", "))
        except Exception:
            try:
                pred_j = json.loads(json_string.replace("'", "\"").replace("} \n{",", ").replace("}; {",", "))
            except Exception:
                try:
                    pred_j = json.loads(json_string.replace('"', '&&&').replace("'", "\"").replace("} \n{",", ").replace("}; {",", ").replace('&&&', "'"))
                except Exception:
                    pred_j = {}
    return pred_j

def create_save_folders(output_folder_path):
    # os.makedirs(f"{output_folder}/{dataset}/{folder_path}")
    os.makedirs(f"{output_folder_path}/")
    os.makedirs(f"{output_folder_path}/prompts/")
    os.makedirs(f"{output_folder_path}/preds/")
    os.makedirs(f"{output_folder_path}/evaluation/")

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

def save_txt_file(file_name, output):
    with open(file_name, 'a') as file:
        for element in output:
            file.write(element+'\n')

def write_in_txt_file(file_name, output):
    with open(file_name, 'a') as file:
        file.write(output)

def load_txt_file(file_name):
    f = open(file_name, 'r')
    t = [line.split('\n')[0] for line in f.readlines()]
    return t

def textada_embeddings(text):
    # Embed some text with text-ada
    model_name = 'text-embedding-ada-002'
    embed = OpenAIEmbeddings(model=model_name, openai_api_key=OPENAI_API_KEY)

    return embed.embed_documents(text)

def top_10(index, dataset_examples, second_embeddings):
    cos = cosine_similarity([dataset_examples[index]], second_embeddings)
    
    # Arrange cosine similarity in dictionary
    cos_dict = {}
    for j, c in enumerate(cos[0]):
        cos_dict[j] = c
    
    # Sort dictionary
    sorted_cos_dict = {k: v for k, v in sorted(cos_dict.items(), key=lambda item: item[1])}
    
    # Retrieve the 10 most similar indices for each test example
    return list(sorted_cos_dict.keys())

#Function to return a flattened list
def flatten_list(original_list):
    flat_list = []
    for item in original_list:
        if isinstance(item, list):
            flat_list = flat_list + item
        else:
            flat_list.append(item)
    return flat_list

# Load data
def load_cta_dataset(dataset, headers):
    # Import labels to text
    if dataset not in ["efthymiou", "limayeu", "limaye", "t2dv2-colnet"]:
        labels_to_text = sienna.load(f"data/labels_to_text_{dataset}-cta.json")
        all_labels = load_txt_file(f"data/{dataset}-labels/{dataset}_all_labels.txt")

        for key in list(labels_to_text.keys()):
            if key not in all_labels:
                del labels_to_text[key]
    else:
        all_labels = load_txt_file(f"data/{dataset}-labels/{dataset}_all_labels.txt")
        labels_to_text = {label: label for label in all_labels}
             
    # Load datasets
    train = load_pickle_file(f'data/{dataset}-cta-train{headers}.pkl')
    if f"{dataset}-cta-val.pkl" in os.listdir("data/"):
        val = load_pickle_file(f'data/{dataset}-cta-val.pkl')
    else:
        val = load_pickle_file(f'data/{dataset}-cta-train{headers}.pkl')
    test = load_pickle_file(f'data/{dataset}-cta-test.pkl')
    
    text_to_label = {labels_to_text[label]: label for label in labels_to_text}
    labels_joined = ", ".join([labels_to_text[l] for l in labels_to_text])

    examples = [example[1] for example in test ]
    labels = [l for example in test for l in example[2]] # In case of multi-label the elements are lists

    val_examples = [example[1] for example in val ]

    val_labels = []
    for table in val:
        data_labels = [ [labels_to_text[label] for label in table[2] if label !=""]] if all(isinstance(ele, str) for ele in table[2]) else [[ [ labels_to_text[label] for label in column_labels] for column_labels in table[2] if column_labels !=""]]
        in_json = pd.DataFrame(data_labels, columns=[table[8][m] for m, l in enumerate(table[2]) if l!=""]).to_json(orient="records").replace("[{","{").replace("}]", "}")
        # pdb.set_trace()
        val_labels.append(in_json)

    train_examples = [ example[1] for example in train ]
    train_labels = []
    for table in train:
        data_labels = [ [labels_to_text[label] for label in table[2] if label !=""]] if all(isinstance(ele, str) for ele in table[2]) else [[ [ labels_to_text[label] for label in column_labels] for column_labels in table[2] if column_labels !=""]]
        in_json = pd.DataFrame(data_labels, columns=[table[8][m] for m, l in enumerate(table[2]) if l!=""]).to_json(orient="records").replace("[{","{").replace("}]", "}")
        train_labels.append(in_json)
    
    return examples, labels, train_examples, train_labels, val_examples, val_labels, labels_to_text, text_to_label, labels_joined, train, val, test

def clean_text(text):
        
    if(isinstance(text, dict)):
        text = '; '.join([ clean_text(v) for k, v in text.items()] )
    elif(isinstance(text, list)):
        text = map(clean_text, text)
        text = '; '.join(text)
        
    if pd.isnull(text):
        return ''
    
    #Remove excess whitespaces
    text = re.sub(' +', ' ', str(text)).strip()
    
    return text