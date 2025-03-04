import pdb
from utils import load_pickle_file, save_pickle_file, load_cta_dataset, create_save_folders
from evaluation_utils import evaluate_table
import os
from dotenv import dotenv_values
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
import tqdm
import sienna


if __name__ == "__main__":
    # Load env file with API KEY using full path
    config = dotenv_values("/full/path/to/file/key.env")
    os.environ['OPENAI_API_KEY'] = config["OPENAI_API_KEY"]
    OPENAI_API_KEY = config["OPENAI_API_KEY"]

    datasets = ["sotabv2-subsetu"] # Possible datasets: "wikitables-2", "limayeu", "sotabv2-subsetu" 
    models = ["gpt-4o-mini"] # Other models: "gpt-4o-2024-05-13"
    methods = [""] # Possible outputs: "-explanation" (model gives an explanation along the annotation), "" (model answers only with annotation)
    run_val = False # Run classification on validation or test set
    output_folder = "output_val" if run_val else "output"
    temperature = 0

    for model_name in models:
        print(model_name)
        chat = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=temperature, model=model_name)

        for dataset in datasets:
            print(dataset)

            dataset_version = "-random-20" if dataset!="limayeu" else ""
            suff = f"-self-cons-{temperature}" if temperature != 0 else ""
            suff += "-hier" if "wikitables" in dataset else ""

            model_path = model_name # to change model path if fine-tuned models, to make path shorter

            # Load dataset
            if not run_val:
                examples, labels, train_examples, train_labels, val_examples, val_labels, labels_to_text, text_to_label, labels_joined, train, val, test = load_cta_dataset(dataset, dataset_version)
            else:
                _, _, train_examples, train_labels, examples, labels, labels_to_text, text_to_label, labels_joined, train, test, _ = load_cta_dataset(dataset, dataset_version)
            
            # Load similar demonstrations
            similar_demos = load_pickle_file(f"embeddings/examples_demonstrations_{dataset}{dataset_version}_test.pkl")

            # Run model X times
            for run in range(0,1):

                # If path doesn't exist: create
                if not os.path.exists(f"{output_folder}/{dataset}/{model_path}"):
                    create_save_folders(f"{output_folder}/{dataset}/{model_path}")

                for method in methods:
                    first_messages = []

                    # Task description and task instructions
                    if dataset in ["wikitables-2","limayeu"]:
                        first_messages.append(SystemMessage(content=f"You are a world-class data engineer and your task is to annotate the columns of a given table with one or more of the following labels that are separated with comma: {labels_joined}."))
                        if dataset in ["wikitables-2"] and "hier" in suff:
                            hier = sienna.load("data/wikitables-hierarchy-v2.json")
                            first_messages.append(SystemMessage(content=f"The labels follow this hierarchy from lowest to highest: {hier}.\n When choosing labels, if the labels are in the lowest hierarchy, choose the higher hierarchies as well."))
                    else:
                        first_messages.append(SystemMessage(content=f"You are a world-class data engineer and your task is to annotate the columns of a given table with only one of the following labels that are separated with comma: {labels_joined}."))
                    
                    if method == "":
                        if dataset in ["wikitables-2","limayeu"]:
                            first_messages.append(SystemMessage(content="Your instructions are: 1. Look at the cell values in detail. The first row of the table corresponds to the column names. 2. For each column, select one or more label/s that best represents the meaning of all cells in the column. The column can have multiple labels that have the same semantic meaning. 3. Answer with the selected label/s for each column using the JSON format {column_name: [label/s]}. 4. Answer only with labels from the provided label set!"))
                        else:
                            first_messages.append(SystemMessage(content="Your instructions are: 1. Look at the cell values in detail. The first row of the table corresponds to the column names. 2. For each column, select a label that best represents the meaning of all cells in the column. 3. Answer with the selected label for each column using the JSON format {column_name: label}. 4. Answer only with labels from the provided label set!"))
                    else:
                        first_messages.append(SystemMessage(content="Your instructions are: 1. Look at the cell values in detail. The first row of the table corresponds to the column names. 2. For each column, select a label that best represents the meaning of all cells in the column. 3. Answer with the selected label and an explanation why the label was chosen for each column using the JSON format {'column_name': ['label', 'explanation']}. 4. Answer only with labels from the provided label set!"))

                    # Zero-shot and Few-shot similar demonstrations:
                    for nr in [0]: # Choose number of demonstrations: 0, 5
                        preds = []
                        messages_list = []

                        for j, example in tqdm.tqdm(enumerate(examples), total=len(examples)):
                            messages = []

                            for message in first_messages:
                                messages.append(message)

                            if nr:
                                # Training demonstrations
                                for index in similar_demos[j][-nr:]:
                                    train_extra_message = "Classify these table columns:" 
                                    if dataset in ["sotabv2-subsetu", "limayeu"]:
                                        train_extra_message = f"Classify {', '.join([f'Column {col_idx+1}' for col_idx, l in enumerate(train[index][2]) if l!=''])}:" # asking for particular columns to be annotated
                                    messages.append(HumanMessage(content=f"{train_extra_message}\n{train_examples[index]}"))
                                    messages.append(AIMessage(content=f"{train_labels[index]}"))

                            # Test table
                            extra_message = "Classify these table columns:"
                            if dataset in ["sotabv2-subsetu", "limayeu"]:
                                extra_message = f"Classify {', '.join([f'Column {col_idx+1}' for col_idx, l in enumerate(test[j][2]) if l!=''])}:"
                            messages.append(HumanMessage(content=f"{extra_message}\n{example}"))
                            
                            # pdb.set_trace()
                            
                            res = chat(messages)
                            preds.append(res)
                            messages_list.append(messages)
                        
                        save_pickle_file(f"{output_folder}/{dataset}/{model_path}/preds/table-{nr}{'-similar' if nr!=0 else ''}-shot{method}{suff}-run_{run}.pkl", preds)
                        save_pickle_file(f"{output_folder}/{dataset}/{model_path}/messages/table-{nr}{'-similar' if nr!=0 else ''}-shot{method}{suff}-run_{run}-messages.pkl", messages_list)
                        evaluate_table(f"{output_folder}/{dataset}/{model_path}/evaluation/table-{nr}{'-similar' if nr!=0 else ''}-shot{method}{suff}-run_{run}", [p.content for p in preds], test, text_to_label)