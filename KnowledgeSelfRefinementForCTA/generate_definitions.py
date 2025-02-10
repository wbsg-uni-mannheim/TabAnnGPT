from functools import partial
import multiprocessing
import os
import pdb
import random
from dotenv import dotenv_values
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from utils import load_cta_dataset, load_pickle_file, parse_json, save_pickle_file, textada_embeddings, top_10
import tqdm
import sienna

def save_label_embeds(defs, def_method, all_example_embeddings):
    # Embed label definitions
    label_embeddings = textada_embeddings([defs[label] for label in defs])
    all_labels = [defn for defn in defs]
    save_pickle_file(f"embeddings/label_embeddings-{def_method}.pkl", label_embeddings)
    
    # Choose for each test/val table 10 most similar definitions
    for mode, example_embeddings in zip(["val","test"], all_example_embeddings):
        pool = multiprocessing.Pool(processes=4)
        examples_labels_ind = list(tqdm(pool.imap(partial(top_10, dataset_examples=example_embeddings, second_embeddings=label_embeddings), range(len(example_embeddings))), total=range(len(example_embeddings))))
        pool.close()
        pool.join()
        examples_labels = [ [ all_labels[i] for i in ind[-10:]] for ind in examples_labels_ind]
        save_pickle_file(f"embeddings/examples_labels_{def_method}_{mode}.pkl", examples_labels)

if __name__ == "__main__":
    # Load env file with API KEY using full path
    config = dotenv_values("/full/path/to/file/key.env")
    os.environ['OPENAI_API_KEY'] = config["OPENAI_API_KEY"]
    OPENAI_API_KEY = config["OPENAI_API_KEY"]

    datasets = ["limaye"] # Datasets: "wikitables-2", "sotabv2-subsetu"
    models = ["gpt-4o-2024-05-13"] # Select model for generating definitions, either gpt-4o or fine-tuned model for the fine-tuning scenario
    for model_name in models:
        print(model_name)
        chat = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0, model=model_name)

        for dataset in datasets:
            print(dataset)
            if "ft" in model_name:
                if "mini" in model_name:
                    model_short_name = "ft-gpt-4o-mini"
                else:
                    model_short_name = "ft-gpt-4o"
            else:
                model_short_name = model_name 
            
            # Generate Demonstration Definitions
            if f"{dataset}-{model_short_name}_demonstration_definitions.json" not in os.listdir(f"data/{dataset}-labels/"):
                examples, labels, train_examples, train_labels, val_examples, val_labels, labels_to_text, text_to_label, labels_joined, train, val, test = load_cta_dataset(dataset, "-random-20-kg" if "limaye" not in dataset else "-kg")
                # Load test and validation embeddings
                # If embeddings to not exist create using embed_datasets.py
                val_embeddings = load_pickle_file(f"embeddings/val_embeddings_{dataset}.pkl")
                test_embeddings = load_pickle_file(f"embeddings/test_embeddings_{dataset}.pkl")
                
                if dataset == "wikitables-2":
                    hier = sienna.load("data/wikitables-hierarchy-v2.json")

                generated_defs = {}
                for label in tqdm.tqdm(labels_to_text, total=len(labels_to_text)):
                    generation_messages = []
                    # Task definition
                    generation_messages.append(SystemMessage(content="Your task is to generate definitions about some terms that can help in the task of column type annotation."))
                    
                    # Task instructions
                    if dataset in ["wikitables-2"] and labels_to_text[label] in hier:
                        generation_messages.append(SystemMessage(content="Your instructions are: 1. Look at the tables given to you and the hierarchy information. 2. The first row of each table are the column types of each column. One column can have multiple column types. 3. Look at the statistical and semantic characteristics of the columns. 4. Generate a definition of the term by looking at the whole table and the hierarchy information of the label. How is the label related to the hierarchy and to the other types in the same column? 5. Reply only with the definition.")) # 6.Reply only with knowledge facts not examples.
                        generation_messages.append(SystemMessage(content=f"The hierarchy of the label '{labels_to_text[label]}' from lower to higher is: {' -> '.join(hier[labels_to_text[label]])}"))
                    else:
                        generation_messages.append(SystemMessage(content="Your instructions are: 1. Look at the tables given to you. 2. The first row of each table are the column types of each column. 3. Look at the statistical and semantic characteristics of the columns. 4.Generate a definition of the term by looking at the whole table. 5. Do not generate specific knowledge for each of the columns. 6. Reply only with the definition."))
                    message_string = f"""Generate a definition about the term '{labels_to_text[label]}' using the following examples"""
                    
                    if dataset in ["wikitables-2"] and label in hier:
                        message_string += f""" and the given hierarchy information:\n"""
                    else:
                        message_string += ":\n"

                    for i in range(3):
                        # Randomly select 3 demonstrations:
                        if dataset in ["wikitables-2", "limaye"]:
                            index = random.choice([j for j, e in enumerate(train_labels) if any(labels_to_text[label] in el for el in parse_json(e).values())])
                        else:
                            index = random.choice([j for j, e in enumerate(train_labels) if labels_to_text[label] in parse_json(e).values()])
                        message_string += f"{train_examples[index]}\n\n"

                    message_string = message_string.strip()
                    generation_messages.append(HumanMessage(content=message_string))
                    
                    # pdb.set_trace()
                    res = chat(generation_messages)
                    generated_defs[label] = res.content
                sienna.save(generated_defs, f"data/{dataset}-labels/{dataset}-{model_short_name}_demonstration_definitions.json")
                # Create label embeddings and choose 10 definitions for each test table
                save_label_embeds(generated_defs, f"{dataset}-{model_short_name}_demonstration_definitions", [val_embeddings, test_embeddings])

            # Generate Initial Definitions
            if f"{dataset}-{model_short_name}_initial_definitions.json" not in os.listdir(f"data/{dataset}-labels/"):
                examples, labels, train_examples, train_labels, val_examples, val_labels, labels_to_text, text_to_label, labels_joined, train, val, test = load_cta_dataset(dataset, "-random-20-kg" if "limaye" not in dataset else "-kg")
                # Load test and validation embeddings
                # If embeddings to not exist create using embed_datasets.py
                val_embeddings = load_pickle_file(f"embeddings/val_embeddings_{dataset}.pkl")
                test_embeddings = load_pickle_file(f"embeddings/test_embeddings_{dataset}.pkl")
                
                if dataset == "wikitables-2":
                    hier = sienna.load("data/wikitables-hierarchy-v2.json")
                
                generated_defs = {}
                for label in tqdm.tqdm(labels_to_text, total=len(labels_to_text)):
                    generation_messages = []
                    # Task description
                    generation_messages.append(SystemMessage(content="Your task is to generate definitions about some terms that can help in the task of column type annotation."))                        
                    
                    # Add hierarchy information for wikiturl
                    if dataset in ["wikitables-2"] and labels_to_text[label] in hier:
                        generation_messages.append(SystemMessage(content=f"The hierarchy of the label '{labels_to_text[label]}' from lower to higher is: {' -> '.join(hier[labels_to_text[label]])}"))
                    message_string = f"""Generate a definition about the term '{labels_to_text[label]}'."""

                    message_string = message_string.strip()
                    generation_messages.append(HumanMessage(content=message_string))
                    
                    # pdb.set_trace()
                    res = chat(generation_messages)
                    generated_defs[label] = res.content
                sienna.save(generated_defs, f"data/{dataset}-labels/{dataset}-{model_short_name}_initial_definitions.json")
                # Create label embeddings and choose 10 definitions for each test table
                save_label_embeds(generated_defs, f"{dataset}-{model_short_name}_initial_definitions", [val_embeddings, test_embeddings])

