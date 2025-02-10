import pdb
from utils import load_pickle_file, load_cta_dataset
from evaluation_utils import group_errors, group_errors_multilabel, save_pickle_file
import os
from dotenv import dotenv_values
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import tqdm
import sienna
import random


if __name__ == "__main__":
    # Choose the model whose errors are used to refine the definitions
    # "gpt-4o-mini", "unsloth-Meta-Llama-3.1-8B-Instruct-bnb-4bit", "unsloth-Meta-Llama-3.1-70B-Instruct-bnb-4bit" or fine-tuned models
    model_name = "gpt-4o"
    # Choose dataset
    dataset = "sotabv2-subsetu" # Other datasets: "limayeu", "wikitables-2"
    method = "" # or "-explanation"
    # Definitions to update
    def_method = "-gpt-4o_demonstration" # or "-ft-gpt-4o_demonstration" for demostration definitions generated from fine-tuned models

    model_path=model_name # if fine-tuned model change path

    new_def_method = f"_{model_path.split('/')[1].replace('0.0001_16_10_5020_wd=0.0_all-layers_','').replace('0.0001_8_10_5020_wd=0.0_all-layers_','')}" if "FT" in model_path else f"_{model_path.replace('/','-')}"
    print(new_def_method)

    # Load env file with API KEY using full path
    config = dotenv_values("/full/path/to/file/key.env")
    os.environ['OPENAI_API_KEY'] = config["OPENAI_API_KEY"]
    OPENAI_API_KEY = config["OPENAI_API_KEY"]

    # Load dataset
    _, _, train_examples, train_labels, val_examples, _, labels_to_text, text_to_label, labels_joined, train, val, _ = load_cta_dataset(dataset, "")
    val_labels = [ vel for ve in val for vel in ve[2] ]
    
    # Load definitions to refine
    label_definitions = sienna.load(f"data/{dataset}-labels/{dataset}{def_method}_definitions.json")
    all_labels = [defn for defn in label_definitions]
    label_definitions_string = "\n".join([ f"{labels_to_text[label]}: {label_definitions[label]}" for label in all_labels ])
    
    suff = ""
    if "Llama" in model_name:
        suff = "-similar"
    if "wikitables" in model_name:
        suff += "-hier"
    
    # Load classification from validation set
    classification_preds = load_pickle_file(f"output_val/{dataset}/{model_path}/evaluation/table-0-shot{def_method}{'-defs' if def_method != '' else ''}{method}{suff}-run_0_predictions.pkl")

    # Initialize model for refinement
    updater_model_name = "gpt-4o-2024-05-13" # or the fine-tuned gpt-4o models in the fine-tuning scenario
    chat = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0, model=updater_model_name)

    preds = []
    messages_list = []
    print(f"Loading definitions: data/{dataset}-labels/{dataset}{def_method}_definitions.json")
    definitions_to_update = sienna.load(f"data/{dataset}-labels/{dataset}{def_method}_definitions.json")
    
    # Group the errors into false positives and false negatives
    if dataset not in ["wikitables-2","limayeu"]:
        fn_errors, fp_errors = group_errors(val, val_labels, classification_preds)
    else:
        fn_errors, fp_errors = group_errors_multilabel(val, val_labels, classification_preds)
        hier = sienna.load("data/wikitables-hierarchy-v2.json")

    # For each label, refine the label definition based on the errors
    for label in tqdm.tqdm(fp_errors, total=len(fp_errors)): # Update all labels that have some error
        # Hierarchy information if dataset is wikiturl
        hier_message = f"\n\nThe hierarchy of the label '{labels_to_text[label]}' from lower to higher is: {' -> '.join(hier[labels_to_text[label]])}\n" if dataset in ["wikitables-2"] and labels_to_text[label] in hier else ""
        
        messages = []
        message = f"""You are a model that updates the definitions of labels based on errors that were done by a model.

The previous definition for label '{labels_to_text[label]}' is the following:
{definitions_to_update[label]}{hier_message}

*** Correct way of using label '{labels_to_text[label]}' ***
"""
        for i in range(0,3):
            # Show 3 random correct ways of using label
            if dataset in ["wikitables-2","limayeu"]:
                index = random.choice([j for j, e in enumerate(train) if any(label in el for el in e[2])])
                column_number = [f"{c+1}" for c, col in enumerate(train[index][2]) if label in col][0]
            else:
                index = random.choice([j for j, e in enumerate(train) if label in e[2]])
                column_number = train[index][2].index(label)+1
            
            message += f"{train[index][1]}\n\n"
            message += f"In the table above, Column {column_number} is an example of label '{labels_to_text[label]}'\n\n"

        if label not in fn_errors:
            message += f"""*** Incorrect ways of using the label '{labels_to_text[label]}' ***
"""
        else:
            message += f"""*** Misclassifications of label '{labels_to_text[label]}' ***
"""
        
        # Show errors done in the validation set
        if dataset not in ["wikitables-2","limayeu"]:
            for j, table_index in enumerate(fp_errors[label][0]):
                if j == 0:
                    message += f"{val[table_index][1]}\n"
                elif table_index != fp_errors[label][0][j-1]:
                    message += f"{val[table_index][1]}\n"
                if fp_errors[label][2][j] != "-":
                    if label not in fn_errors:
                        message += f"In the table above, Column {fp_errors[label][1][j]} was wrongly classified as '{labels_to_text[label]}' but the correct label is '{labels_to_text[fp_errors[label][2][j]]}'\n\n"
                    else:
                        message += f"In the table above, Column {fp_errors[label][1][j]} was wrongly classified as '{labels_to_text[fp_errors[label][2][j]]}' but the correct label is '{labels_to_text[label]}'\n\n"
                else:
                    if label not in fn_errors:
                        message += f"In the table above, Column {fp_errors[label][1][j]} was wrongly classified as '{labels_to_text[label]}'\n\n"
                    else:
                        message += f"In the table above, Column {fp_errors[label][1][j]} was wrongly classified as '{labels_to_text[fp_errors[label][2][j]]}'\n\n"
            
        else:
            for j, table_index in enumerate(fp_errors[label][0]): # Loop through table indices where the errors were made
                if j == 0:
                    message += f"{val[table_index][1]}\n"
                elif table_index != fp_errors[label][0][j-1]:
                    message += f"{val[table_index][1]}\n"
                
                correct_labels_for_table = val[table_index][2][fp_errors[label][1][j]-1]

                if all(e =="-" for e in fp_errors[label][2][j]):
                    message += f"In the table above, Column {fp_errors[label][1][j]} was wrongly classified but the {'correct label is' if len(correct_labels_for_table)<2 else 'correct labels are'} '{', '.join([labels_to_text[l] for l in correct_labels_for_table])}'\n\n"
                else:
                    if label not in fn_errors:
                        message += f"In the table above, Column {fp_errors[label][1][j]} was wrongly classified as '{labels_to_text[label]}' but the {'correct label is' if len(fp_errors[label][2][j])<2 else 'correct labels are'} '{', '.join([labels_to_text[l] for l in fp_errors[label][2][j]])}'\n\n"
                    elif label in fn_errors and set(fp_errors[label][2][j]) - set(correct_labels_for_table) == 0:
                        message += f"In the table above, Column {fp_errors[label][1][j]} was classified as '{', '.join([labels_to_text[l] for l in fp_errors[label][2][j]])}' and is missing the label '{labels_to_text[label]}'\n\n"
                    else:
                        message += f"In the table above, Column {fp_errors[label][1][j]} was wrongly classified as '{', '.join([labels_to_text[l] for l in fp_errors[label][2][j] if l not in correct_labels_for_table and l!='-'])}' but the correct label is '{labels_to_text[label]}'\n\n"

        messages.append(SystemMessage(content=message))
        if dataset in ["wikitables-2", "limayeu"]:
            messages.append(HumanMessage(content=f"Update or rewrite the old definition of the '{labels_to_text[label]}' label so that the wrong classifications do not happen again. Delete the sentences that could have lead to the mistake. In the update/rewriting do not include general information about the label, but information on how the label is distinguished based on context or keywords from the other labels. Short examples can be added. Reply only with the new definition!"))
        else:
            messages.append(HumanMessage(content=f"Update or rewrite the old definition of the '{labels_to_text[label]}' label so that the wrong classifications do not happen again. Reply only with the new definition!"))
        
        # pdb.set_trace()

        res = chat(messages)
        preds.append(res)
        messages_list.append(messages)
        definitions_to_update[label] = res.content

        save_pickle_file(f"output_val/{dataset}/{model_path}/preds/table-0-shot{method}{def_method}{new_def_method}{suff}-refine-defs.pkl", preds)
        save_pickle_file(f"output_val/{dataset}/{model_path}/messages/table-0-shot{method}{def_method}{new_def_method}{suff}-refine-defs-messages.pkl", messages_list)
        sienna.save(definitions_to_update, f"data/{dataset}-labels/{dataset}-{new_def_method}_refined_definitions.json")
        