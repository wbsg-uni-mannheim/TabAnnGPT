from utils import create_save_folders, save_pickle_file, load_cta_dataset
from evaluation_utils import evaluate_table
import os
from dotenv import dotenv_values
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import tqdm
import sienna
import pdb


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

    for model_name in models:
        print(model_name)
        chat = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0, model=model_name)

        for dataset in datasets:
            suff="-hier" if "wikitables" in dataset else ""
            dataset_version = "-random-20" if dataset!="limayeu" else ""

            print(dataset)
            model_path = model_name # to change model path if fine-tuned models, to make path shorter

            # Load dataset
            if not run_val:
                examples, labels, train_examples, train_labels, val_examples, val_labels, labels_to_text, text_to_label, labels_joined, train, val, test = load_cta_dataset(dataset, dataset_version)
            else:
                _, _, train_examples, train_labels, examples, labels, labels_to_text, text_to_label, labels_joined, train, test, _ = load_cta_dataset(dataset, dataset_version)
            
            # Name of definitions, examples:
            # "-ft-gpt-4o_demonstration": labels generated from the fine-tuned models = demonstration definitions
            # "-gpt-4o-2024-05-13_demonstration": labels generated from gpt-4o = demonstration definitions
            # "-gpt-4o-mini-comparative": comparative definitions generated for gpt-mini from the errors of the validation set
            # "-gpt-4o-2024-05-13_demonstration_gpt-4o-2024-05-13_refined": refined definitions from the errors of the gpt-4o model on the validation set, used with the gpt-4o model
            for def_method in ["-gpt-4o-2024-05-13_demonstration"]:
                # Load generated definitions
                label_definitions = sienna.load(f"data/{dataset}-labels/{dataset}{def_method}_definitions.json")
                all_labels = [defn for defn in label_definitions]
                # Show all label definitions
                label_definitions_string = "\n".join([ f"{labels_to_text[label]}: {label_definitions[label]}" for label in all_labels if label_definitions[label]!="" ]) # label in set(labels) and 

                for run in range(0,1):
                    # If path doesn't exist: create
                    if not os.path.exists(f"{output_folder}/{dataset}/{model_path}"):
                        create_save_folders(f"{output_folder}/{dataset}/{model_path}")

                    for method in methods:
                        if f"table-0-shot{method}{dataset_version}{def_method}-defs{suff}_{run}.pkl" not in os.listdir(f"{output_folder}/{dataset}/{model_path}/preds/"):

                            preds = []
                            messages_list = []

                            for j, example in tqdm.tqdm(enumerate(examples), total=len(examples)):
                                messages = []

                                # Task description
                                if dataset in ["wikitables-2","limayeu"]:
                                    messages.append(SystemMessage(content=f"You are a world-class data engineer and your task is to annotate the columns of a given table with one or more of the following labels that are separated with comma: {labels_joined}."))
                                    if dataset == "wikitables-2":
                                        hier = sienna.load("data/wikitables-hierarchy-v2.json")
                                        messages.append(SystemMessage(content=f"The labels follow this hierarchy from lowest to highest: {hier}.\n When choosing labels, if the labels are in the lowest hierarchy, choose the higher hierarchies as well."))
                                else:
                                    messages.append(SystemMessage(content=f"You are a world-class data engineer and your task is to annotate the columns of a given table with only one of the following labels that are separated with comma: {labels_joined}."))
                                
                                # Add generated labels to the prompt
                                messages.append(SystemMessage(content=f"The definitions of the labels are the following:\n{label_definitions_string}"))

                                # Task instructions
                                if method == "":
                                    if dataset in ["wikitables-2", "limayeu"]:
                                        messages.append(SystemMessage(content="Your instructions are: 1. Look at the cell values in detail. The first row of the table corresponds to the column names. 2. For each column, select one or more label/s that best represents the meaning of all cells in the column. The column can have multiple labels that have the same semantic meaning. 3. Answer with the selected label/s for each column using the JSON format {column_name: [label/s]}. 4. Answer only with labels from the provided label set!"))
                                    else:
                                        messages.append(SystemMessage(content="Your instructions are: 1. Look at the cell values in detail. The first row of the table corresponds to the column names. 2. For each column, select a label that best represents the meaning of all cells in the column. 3. Answer with the selected label for each column using the JSON format {column_name: label}. 4. Answer only with labels from the provided label set!"))
                                else:
                                    messages.append(SystemMessage(content="Your instructions are: 1. Look at the cell values in detail. The first row of the table corresponds to the column names. 2. For each column, select a label that best represents the meaning of all cells in the column. 3. Answer with the selected label and an explanation why the label was chosen for each column using the JSON format {'column_name': ['label', 'explanation']}. 4. Answer only with labels from the provided label set!"))
                            
                                extra_message = "Classify these table columns:"
                                if dataset in ["sotabv2-subsetu", "limayeu"]:
                                    extra_message = f"Classify {', '.join([f'Column {col_idx+1}' for col_idx, l in enumerate(test[j][2]) if l!=''])}:"
                                messages.append(HumanMessage(content=f"{extra_message}\n{example}"))
                                
                                # pdb.set_trace()
                                
                                res = chat(messages)
                                preds.append(res)
                                messages_list.append(messages)
                            
                            save_pickle_file(f"{output_folder}/{dataset}/{model_path}/preds/table-0-shot{method}{def_method}-defs{suff}-run_{run}.pkl", preds)
                            save_pickle_file(f"{output_folder}/{dataset}/{model_path}/messages/table-0-shot{method}{def_method}-defs{suff}-run_{run}-messages.pkl", messages_list)
                            evaluate_table(f"{output_folder}/{dataset}/{model_path}/evaluation/table-0-shot{method}{def_method}-defs{suff}-run_{run}", [p.content for p in preds], test, text_to_label)
