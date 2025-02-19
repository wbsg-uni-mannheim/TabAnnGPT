import os
from dotenv import dotenv_values
import tqdm
from langchain_community.embeddings import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import multiprocessing
import tqdm
import pdb
import sys
# setting path
sys.path.append('..') # or full path to KnowledgeSelfRefinementForCTA folder
from utils import save_pickle_file, load_cta_dataset, load_pickle_file

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

if __name__ == "__main__":

    # Load env file with API KEY using full path
    config = dotenv_values("/full/path/to/file/key.env")
    os.environ['OPENAI_API_KEY'] = config["OPENAI_API_KEY"]
    OPENAI_API_KEY = config["OPENAI_API_KEY"]

    # Load embedding model
    model_name = 'text-embedding-3-small'
    embed = OpenAIEmbeddings(model=model_name, openai_api_key=OPENAI_API_KEY)

    # CTA datasets
    datasets = [
        "sotabv2-subset",
        "wikitables-2",
        "limaye",
    ]
    
    for dataset in datasets:
        print(dataset)
        dataset_version = "-random-20" if "limaye" not in dataset else ""
        # Load dataset
        examples, _, train_examples, _, val_examples, _, _, _, _, train, _, _ = load_cta_dataset(dataset,dataset_version)
        
        # Create embeddings for training, validation and text examples
        if f"train_embeddings_{dataset}{dataset_version}.pkl" not in os.listdir(f"embeddings/"):
            train_embeddings = embed.embed_documents(train_examples)
            save_pickle_file(f"embeddings/train_embeddings_{dataset}{dataset_version}.pkl", train_embeddings)
        else:
            train_embeddings = load_pickle_file(f"embeddings/train_embeddings_{dataset}{dataset_version}.pkl")
        
        if len(val_examples) and f"val_embeddings_{dataset}.pkl" not in os.listdir(f"embeddings/"):
            val_embeddings = embed.embed_documents(val_examples)
            save_pickle_file(f"embeddings/val_embeddings_{dataset}.pkl", val_embeddings)
        elif f"val_embeddings_{dataset}.pkl" in os.listdir(f"embeddings/"):
            val_embeddings = load_pickle_file(f"embeddings/val_embeddings_{dataset}.pkl")
        
        if f"test_embeddings_{dataset}.pkl" not in os.listdir(f"embeddings/"):
            test_embeddings = embed.embed_documents(examples)
            save_pickle_file(f"embeddings/test_embeddings_{dataset}.pkl", test_embeddings)
        else:
            test_embeddings = load_pickle_file(f"embeddings/test_embeddings_{dataset}.pkl")
 
        
        if f"examples_demonstrations_{dataset}{dataset_version}_test.pkl" not in os.listdir(f"embeddings/"):
            # Retrieve top 10 indices for each test label
            pool = multiprocessing.Pool(processes=4)
            examples_demonstrations = list(tqdm.tqdm(pool.imap(top_10_indices, range(len(examples))), total=len(examples)))
            pool.close()
            pool.join()

            # Save most similar training examples to test examples
            save_pickle_file(f"embeddings/examples_demonstrations_{dataset}{dataset_version}_test.pkl", examples_demonstrations)
        