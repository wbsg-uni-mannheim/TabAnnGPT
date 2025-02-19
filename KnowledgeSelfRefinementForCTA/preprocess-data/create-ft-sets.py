import os
import pdb
import random
import tqdm
import sienna
import sys
# setting path
sys.path.append('..') # or full path to KnowledgeSelfRefinementForCTA folder
from utils import load_cta_dataset, parse_json

# Definition as explanation with prefix Label [label] is correct because [label definition]
def get_explanation(train_lab, lab_defs):
    train_lab = eval(train_lab)
    train_lab_explanation = {}

    for col in train_lab:
        if isinstance(train_lab[col], str): # Multi-class
            train_lab_explanation[col] = [train_lab[col], f"Label {train_lab[col]} is correct because {lab_defs[text_to_label[train_lab[col]]]}"]
        else: # Multi-label
            train_lab_explanation[col] = [train_lab[col], f"Label {' and '.join(train_lab[col])} is correct because " + " and ".join([f"{lab_defs[text_to_label[c]]}" for c in train_lab[col]])]

    return train_lab_explanation

if __name__ == "__main__":
    datasets = ["sotabv2-subsetu"] # Other dataset: "wikitables-2"
    dataset_version = "-random-20"
    # Methods:
    # "" => empty string for simple set without instructions
    # "-with" => fine tuning with definitions set
    # "-multi-task" => multi-task set without instructions and without demonstrations
    # "-multi-task-3" => multi-task set without instructions and 3 random demonstrations for the definition generation set
    method = "-instr-multi-task"

    for file in datasets:
        for def_method in [""]: # Empty for simple-tuning, otherwise demonstration definitions: "-gpt-4o-2024-05-13_demonstration"
            print(file)
            # Load dataset
            examples, labels, train_examples, train_labels, val_examples, val_labels, labels_to_text, text_to_label, labels_joined, train, val, test = load_cta_dataset(file, dataset_version)

            pdb.set_trace()
            if file in ["wikitables-2","limayeu"]:
                system = f"You are a world-class data engineer and your task is to annotate the columns of a given table with one or more of the following labels that are separated with comma: {labels_joined}." + " Reply only with the selected label/s for each column using the JSON format {column_name: [label/s]}."
            else:
                system = f"You are a world-class data engineer and your task is to annotate the columns of a given table with only one of the following labels that are separated with comma: {labels_joined}." + " Reply only with the JSON format {column_name: label}."

            # Load definitions
            if def_method:
                label_definitions = sienna.load(f"data/{file}-labels/{file}{def_method}_definitions.json")
                for defn in label_definitions:
                    if "**:" in label_definitions[defn]:
                        split_by_form = label_definitions[defn].split("**:")
                        label_definitions[defn] = "".join(split_by_form[1:]).replace(defn, labels_to_text[defn]).strip()
            
            # Creating Train set
            train_json = {"data":[]}
            
            for j, table in enumerate(train_examples):
                extra_message = "Classify these table columns:"
                if file == "sotabv2-subsetu" or file == "limayeu":
                    extra_message = f"Classify {', '.join([f'Column {col_idx+1}' for col_idx, l in enumerate(train[j][2]) if l!=''])}:"

                assistant_mess = f"{train_labels[j]}"
                if "with" in method:
                    assistant_mess = f"{get_explanation(train_labels[j], label_definitions)}"
                
                train_json["data"].append({
                "instruction": system, # Task description
                "user": f"{extra_message}\n{table}", # Input table
                "assistant": assistant_mess # Ground truth
                })

            # Add definition generation set if multi-task set
            if "multi" in method:
                end_nr = 2
                if "-3" in method:
                    end_nr = 4
                for j in range(1,end_nr):
                    for label in labels_to_text:
                        instructions = "Your task is to generate definitions about some terms that can help in the task of column type annotation."
                        user_mess = f"Generate a definition about the term '{labels_to_text[label]}'."

                        # Add 3 random demontsrations for multi-task-3 set
                        if "-3" in method:
                            user_mess = f"Generate a definition about the term '{labels_to_text[label]}' using the following examples:\n"

                            for i in range(3):
                                # Randomly select 3 examples:
                                if file in ["wikitables-2"]:
                                    index = random.choice([j for j, e in enumerate(train_labels) if any(labels_to_text[label] in el for el in parse_json(e).values())])
                                else:
                                    index = random.choice([j for j, e in enumerate(train_labels) if labels_to_text[label] in parse_json(e).values()])
                                user_mess += f"{train_examples[index]}\n\n"
                            user_mess = user_mess.strip()

                        train_json["data"].append({
                        "instruction": instructions, # Task instructions
                        "user": user_mess, # Task input
                        "assistant": f"{label_definitions[label].replace(label, labels_to_text[label])}" # Ground truth
                        })

            sienna.save(train_json, f"data/sets/{file}-cta-train{dataset_version}{method}{def_method}.json")
            
            # Creating Validation set
            if f"{file}-cta-val.json" not in os.listdir("data/sets/"):
                val_json = {"data":[]}

                for j, table in tqdm.tqdm(enumerate(val_examples), total=len(val_examples)):
                    extra_message = "Classify these table columns:"
                    if file == "sotabv2-subsetu" or file == "limayeu":
                        extra_message = f"Classify {', '.join([f'Column {col_idx+1}' for col_idx, l in enumerate(val[j][2]) if l!=''])}:"

                    assistant_mess = f"{val_labels[j]}"
                    if "with" in method:
                        assistant_mess = f"{get_explanation(val_labels[j], label_definitions)}"
                
                    val_json["data"].append({
                    "instruction": system,
                    "user": f"{extra_message}\n{table}",
                    "assistant": assistant_mess # Ground truth
                    })

                if "with" in method:
                    sienna.save(val_json, f"data/sets/{file}-cta-val{method}{def_method}.json")
                else:
                    sienna.save(val_json, f"data/sets/{file}-cta-val.json")
