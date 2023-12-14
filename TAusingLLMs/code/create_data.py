import os
import json
import numpy as np
from dotenv import dotenv_values
from utils import save_pickle_file, load_cta_dataset, load_cta_dataset_column, load_cpa_dataset, load_cpa_dataset_column
import tqdm
from langchain.embeddings.openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import multiprocessing
import tqdm
from itertools import product

def top_10_indices(index):
    cos = cosine_similarity([test_embeddings[index]], train_embeddings)
    
    # Arrange cosine similarity in dictionary
    cos_dict = {}
    for j, c in enumerate(cos[0]):
        cos_dict[j] = c
    
    # Sort dictionary
    sorted_cos_dict = {k: v for k, v in sorted(cos_dict.items(), key=lambda item: item[1])}
    
    # Retrieve the 10 most similar indices for each test example
    return list(sorted_cos_dict.keys())[-10:]

def calculate_nonoverlap(index1, index2):
    if index1 == index2:
        return 0
    elif index2>index1:
        return len(set(train[index1][2])-set(train[index2][2])) + len(set(train[index2][2])-set(train[index1][2]))

def find_most_similar_with_diff_labels(train_index):
    # Arrange cosine similarity in dictionary
    train_simdisim_dict = {}
    for j, c in enumerate(training_similarity_matrix[train_index]):
        train_simdisim_dict[j] = c

    # Sort dictionary
    sorted_train_simdisim_dict = {k: v for k, v in sorted(train_simdisim_dict.items(), reverse=True, key=lambda item: item[1])}

    selected_index = 0

    if len([key for key in list(sorted_train_simdisim_dict.keys())[:20] if table[train_index][key] > 1]) == 0:
        #smth
        # print("in")
        for key in sorted_train_simdisim_dict:
            if table[train_index][key] > 1:
                selected_index = key
                break
    else:
        #normal way
        max = 0
        id = 0
        for key in list(sorted_train_simdisim_dict.keys())[:20]:
            if table[train_index][key] > max:
                id = key
        selected_index = id

    return selected_index

if __name__ == "__main__":

    # Load env file with API KEY using full path
    config = dotenv_values("/full/path/to/file/key.env")
    os.environ['OPENAI_API_KEY'] = config["OPENAI_API_KEY"]
    OPENAI_API_KEY = config["OPENAI_API_KEY"]

    # Load embedding model
    model_name = 'text-embedding-ada-002'
    embed = OpenAIEmbeddings(model=model_name, openai_api_key=OPENAI_API_KEY)

    # CTA datasets
    datasets = ["sotabv2", "t2dv2-webtables", "sportstables"]

    for format_ in ["", "-column"]:
        for dataset in datasets[:2]:
            print(dataset)
            # Load dataset
            if format_ != "":
                examples, labels, train_examples, train_labels, labels_to_text, text_to_label, labels_joined, train, test = load_cta_dataset_column(dataset,"")
            else:
                examples, labels, test_table_type_labels, train_examples, train_example_labels, train_table_type_labels, labels_to_text, text_to_label, labels_joined, train, test = load_cta_dataset(dataset,"")
    
            # Create embeddings for training and text examples
            test_embeddings = embed.embed_documents(examples)
            train_embeddings = embed.embed_documents(train_examples)
            
            # Save embeddings
            save_pickle_file(f"embeddings/train_embeddings_{dataset}{format_}.pkl", train_embeddings)
            save_pickle_file(f"embeddings/test_embeddings_{dataset}{format_}.pkl", test_embeddings)

            # Retrieve top 10 indices for each test label
            pool = multiprocessing.Pool(processes=4)
            examples_demonstrations = list(tqdm.tqdm(pool.imap(top_10_indices, range(len(test))), total=len(test)))
            pool.close()
            pool.join()
            # Save most similar training examples to test examples
            save_pickle_file(f"embeddings/examples_demonstrations_{dataset}{format_}.pkl", examples_demonstrations)

            # For cc-demos
            training_similarity_matrix = cosine_similarity(train_embeddings, train_embeddings)

            pool = multiprocessing.Pool(processes=20)
            res = pool.starmap(calculate_nonoverlap, product(range(len(train)), repeat=2))
            pool.close()
            pool.join()

            table = np.zeros(shape=(len(train),len(train)))
            for i in range(len(train)):
                count=0
                for row in res[i*len(train):i*len(train)+len(train)]:
                    value = row
                    if value is None:
                        table[i][count] = 0
                    else:
                        table[i][count] = value
                    count += 1

            for i in range(len(train)):
                for j in range(len(train)):
                    if(j<i):
                        table[i][j] = table[j][i]

            # For each test table i, retrieve the two most similar train tables and find the less overlapping tables to the train tables in the training set
            cc_demonstrations = []
            for i in range(len(examples)):
                cc_indices = []
                for index in examples_demonstrations[i][-2:]:
                    cc_indices.append(index)
                    other_index = find_most_similar_with_diff_labels(index)
                    cc_indices.append(other_index)
                cc_demonstrations.append(cc_indices)
            save_pickle_file(f"embeddings/cc_examples_demonstrations_{dataset}{format_}.pkl", cc_demonstrations)

            # Manual definitions
            f = open(f'../data/{dataset}-definitions.txt')
            definitions = json.load(f)
            definitions = [definitions[defn] for defn in definitions]
            
            definitions_embeddings = embed.embed_documents(definitions)
            save_pickle_file(f"embeddings/{dataset}-definitions-embeddings.pkl", definitions_embeddings)
            
    

    # CPA datasets
    datasets = ["sotabv2", "t2dv2-webtables"]

    for format_ in ["", "-column"]:
        for dataset in datasets:
            print(dataset)

            # Load datasets
            if format_ != "":
                examples, labels, train_examples, train_labels, labels_to_text, text_to_label, labels_joined, train, test = load_cpa_dataset_column(dataset,"")
            else:
                examples, labels, test_table_type_labels, train_examples, train_example_labels, train_table_type_labels, labels_to_text, text_to_label, labels_joined, train, test = load_cpa_dataset(dataset,"",False)

            test_embeddings = embed.embed_documents(examples)
            train_embeddings = embed.embed_documents(train_examples)
            
            # Save embeddings
            save_pickle_file(f"embeddings/cpa-train_embeddings_{dataset}{format_}.pkl", train_embeddings)
            save_pickle_file(f"embeddings/cpa-test_embeddings_{dataset}{format_}.pkl", test_embeddings)

            # Retrieve top 10 indices for each test label
            pool = multiprocessing.Pool(processes=4)
            examples_demonstrations = list(tqdm.tqdm(pool.imap(top_10_indices, range(len(test))), total=len(test)))
            pool.close()
            pool.join()
            # Save most similar training examples to test examples
            save_pickle_file(f"embeddings/cpa-examples_demonstrations_{dataset}{format_}.pkl", examples_demonstrations)

            # For cc-demos
            training_similarity_matrix = cosine_similarity(train_embeddings, train_embeddings)

            pool = multiprocessing.Pool(processes=20)
            res = pool.starmap(calculate_nonoverlap, product(range(len(train)), repeat=2))
            pool.close()
            pool.join()

            table = np.zeros(shape=(len(train),len(train)))
            for i in range(len(train)):
                count=0
                for row in res[i*len(train):i*len(train)+len(train)]:
                    value = row
                    if value is None:
                        table[i][count] = 0
                    else:
                        table[i][count] = value
                    count += 1

            for i in range(len(train)):
                for j in range(len(train)):
                    if(j<i):
                        table[i][j] = table[j][i]

            # For each test table i, retrieve the two most similar train tables and find the less overlapping tables to the train tables in the training set
            cc_demonstrations = []
            for i in range(len(examples)):
                cc_indices = []
                for index in examples_demonstrations[i][-2:]:
                    cc_indices.append(index)
                    other_index = find_most_similar_with_diff_labels(index)
                    cc_indices.append(other_index)
                cc_demonstrations.append(cc_indices)
            save_pickle_file(f"embeddings/cpa-cc_examples_demonstrations_{dataset}{format_}.pkl", cc_demonstrations)

            # Manual definitions
            f = open(f'../data/cpa-{dataset}-definitions.txt')
            definitions = json.load(f)
            definitions = [definitions[defn] for defn in definitions]
            
            definitions_embeddings = embed.embed_documents(definitions)
            save_pickle_file(f"embeddings/cpa-{dataset}-definitions-embeddings.pkl", definitions_embeddings)
            





