import os
import pdb
import random
import tqdm
import sienna
import sys
# setting path
sys.path.append('..') # or full path to KnowledgeSelfRefinementForCTA folder
from utils import load_cta_dataset, parse_json

if __name__ == "__main__":
    datasets = ["sotabv2-subsetu"] # Other dataset: "wikitables-2"
    dataset_version = "-random-20"
    # Methods:
    # "-instr" => simple set with instructions
    # "" => empty string for simple set without instructions
    # "-instr-multi-task" => multi-task set with instructions and without demonstrations
    # "-multi-task" => multi-task set without instructions and without demonstrations
    method = "-instr-multi-task"

    for file in datasets:
        for def_method in ["-gpt-4o-2024-05-13_demonstration"]:
            print(file)
            # Load dataset
            examples, labels, train_examples, train_labels, val_examples, val_labels, labels_to_text, text_to_label, labels_joined, train, val, test = load_cta_dataset(file, dataset_version)

            pdb.set_trace()
            if file in ["wikitables-2","limayeu"]:
                system = f"You are a world-class data engineer and your task is to annotate the columns of a given table with one or more of the following labels that are separated with comma: {labels_joined}." + " Reply only with the selected label/s for each column using the JSON format {column_name: [label/s]}."
                instructions = "Your instructions are: 1. Look at the cell values in detail. The first row of the table corresponds to the column names. 2. For each column, select one or more label/s that best represents the meaning of all cells in the column. The column can have multiple labels that have the same semantic meaning. 3. Answer with the selected label/s for each column using the JSON format {column_name: [label/s]}. 4. Answer only with labels from the provided label set!"
            else:
                system = f"You are a world-class data engineer and your task is to annotate the columns of a given table with only one of the following labels that are separated with comma: {labels_joined}." + " Reply only with the JSON format {column_name: label}."
                instructions = "Your instructions are: 1. Look at the cell values in detail. The first row of the table corresponds to the column names. 2. For each column, select a label that best represents the meaning of all cells in the column. 3. Answer with the selected label for each column using the JSON format {column_name: label}. 4. Answer only with labels from the provided label set!"

            # Load definitions
            if def_method:
                label_definitions = sienna.load(f"data/{file}-labels/{file}{def_method}_definitions.json")
                for defn in label_definitions:
                    if "**:" in label_definitions[defn]:
                        split_by_form = label_definitions[defn].split("**:")
                        label_definitions[defn] = "".join(split_by_form[1:]).replace(defn, labels_to_text[defn]).strip()
            
            # Creating Train set
            train_ft_dataset = []
            
            for j, example in tqdm.tqdm(enumerate(train_examples), total=len(train_examples)):
                extra_message = "Classify these table columns:"
                if file == "sotabv2-subsetu" or file == "limayeu":
                    extra_message = f"Classify {', '.join([f'Column {col_idx+1}' for col_idx, l in enumerate(train[j][2]) if l!=''])}:"
                assistant_mess = train_labels[j]

                messages = []
                # Task description
                messages.append({"role": "system", "content": system})
                # Task instructions
                if "instr" in method:
                    messages.append({"role": "system", "content": instructions})
                # Input table
                messages.append({"role": "user", "content": f"{extra_message}\n{example}"})
                # Ground truth
                messages.append({"role": "assistant", "content": f'{assistant_mess}'})
                
                train_ft_dataset.append({"messages":messages})

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

                        train_ft_dataset.append({
                            "messages":[
                                {"role": "system", "content": instructions},
                                {"role": "user", "content": f"{user_mess}"},
                                {"role": "assistant", "content": f"{label_definitions[label].replace(label, labels_to_text[label])}"}
                            ]
                        })
            
            # pdb.set_trace()
            sienna.save(train_ft_dataset, f"data/openai-ft-sets/{file}-cta-train{dataset_version}{method}{def_method}.jsonl")

            # Validation set
            if f"{file}-cta-val{'-instr' if 'instr' in method else ''}.jsonl" not in os.listdir("data/openai-ft-sets/"):
                val_ft_dataset = []
                for j, example in tqdm.tqdm(enumerate(val_examples), total=len(val_examples)):
                    extra_message = "Classify these table columns:"
                    if file == "sotabv2-subsetu" or file == "limayeu":
                        extra_message = f"Classify {', '.join([f'Column {col_idx+1}' for col_idx, l in enumerate(val[j][2]) if l!=''])}:"
                    assistant_mess = val_labels[j]
                    
                    messages = []
                    # Task description
                    messages.append({"role": "system", "content": system})
                    # Task instructions
                    if "instr" in method:
                        messages.append({"role": "system", "content": instructions})
                    # Input table
                    messages.append({"role": "user", "content": f"{extra_message}\n{example}"})
                    # Ground truth
                    messages.append({"role": "assistant", "content": f'{assistant_mess}'})

                    val_ft_dataset.append({"messages":messages})

                
                if "with" in method:
                    sienna.save(val_ft_dataset, f"data/openai-ft-sets/{file}-cta-val{'-instr' if 'instr' in method else ''}{method}{def_method}.jsonl")
                else:
                    sienna.save(val_ft_dataset, f"data/openai-ft-sets/{file}-cta-val{'-instr' if 'instr' in method else ''}.jsonl")
