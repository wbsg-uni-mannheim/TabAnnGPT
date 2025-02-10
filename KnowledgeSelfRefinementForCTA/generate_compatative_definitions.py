import pdb
from utils import load_pickle_file, save_pickle_file, load_cta_dataset, parse_json
from evaluation_utils import evaluate_table, group_errors, group_errors_multilabel
import os
from dotenv import dotenv_values
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import tqdm
import sienna
import random


if __name__ == "__main__":
    
    dataset = "wikitables-2" # Other datasets: "sotabv2-subsetu", "limayeu"
    suff = "-hier" if "wikitables" in dataset else ""

    # Models to generate comparative defs for
    models = [
        # "gpt-4o-2024-05-13", 
        # "gpt-4o-mini", 
        # "unsloth-Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        # "unsloth-Meta-Llama-3.1-70B-Instruct-bnb-4bit",
    ]

    # Load env file with API KEY using full path
    config = dotenv_values("/full/path/to/file/key.env")
    os.environ['OPENAI_API_KEY'] = config["OPENAI_API_KEY"]
    OPENAI_API_KEY = config["OPENAI_API_KEY"]

    # Initialize generation model
    generation_model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0, model="gpt-4o-2024-05-13") # for fine-tuning scenario with knowledge prompting, change to fine-tuned model name

    for model_name in models:
        # Load dataset
        _, _, train_examples, train_labels, val_examples, _, labels_to_text, text_to_label, labels_joined, train, val, _ = load_cta_dataset(dataset, "-random-20" if dataset!="limayeu" else "")
        val_labels = [ vel for ve in val for vel in ve[2] ]

        # Load the zero-shot prompting predictions of the validation set
        predictions_before = load_pickle_file(f"output_val/{dataset}/{model_name}/evaluation/table-0-shot{suff}-run_0_predictions.pkl")
        
        # Define model path where to save results, change for ft models
        model_path = model_name

        # Group validation errors
        if dataset not in ["wikitables-2", "limayeu"]:
            fn_errors, fp_errors = group_errors(val, val_labels, predictions_before)
        else:
            fn_errors, fp_errors = group_errors_multilabel(val, val_labels, predictions_before)
            hier = sienna.load("data/wikitables-hierarchy-v2.json")

        # Generate comparative definitions
        preds = []
        messages_list = []
        error_feedback = {}

        for label in tqdm.tqdm(fp_errors, total=len(fp_errors)):
            hier_message = f"\nThe hierarchy of the label '{labels_to_text[label]}' from lower to higher is: {' -> '.join([labels_to_text[l] for l in hier[label]])}\n\n" if dataset in ["wikitables-2"] and label in hier else ""
            
            messages = []
            wrong_labels = set()

            message = f"""You are a model that gives tips on how to distinguish labels from each other based on some examples given to you.
{hier_message}*** Correct way of using label '{labels_to_text[label]}' ***
"""
            for i in range(0,3):
                # Show 3 random correct ways of using label
                if dataset in ["wikitables-2", "limayeu"]:
                    index = random.choice([j for j, e in enumerate(train) if any(label in el for el in e[2])])
                    column_number = [f"{c+1}" for c, col in enumerate(train[index][2]) if label in col][0]
                else:
                    index = random.choice([j for j, e in enumerate(train) if label in e[2]])
                    column_number = train[index][2].index(label)+1

                message += f"{train[index][1]}\n\n"
                message += f"In the table above, Column {column_number} is an example of label '{labels_to_text[label]}'\n\n"

            if label not in fn_errors:#FP
                message += f"""*** Incorrect ways of using the label '{labels_to_text[label]} ***
""" 
            else:#FN
                message += f"""*** Misclassifications of label '{labels_to_text[label]}' ***
"""        

            if dataset not in ["wikitables-2", "limayeu"]:
                for j, table_index in enumerate(fp_errors[label][0]):
                    if j == 0:
                        message += f"{val[table_index][1]}\n"
                    elif table_index != fp_errors[label][0][j-1]:
                        message += f"{val[table_index][1]}\n"
                    if fp_errors[label][2][j] != "-":
                        wrong_labels.add(labels_to_text[fp_errors[label][2][j]])
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
                for j, table_index in enumerate(fp_errors[label][0]): #Loop through table indices where the error was made
                    if j == 0:
                        message += f"{val[table_index][1]}\n"
                    elif table_index != fp_errors[label][0][j-1]:
                        message += f"{val[table_index][1]}\n"
                    
                    correct_labels_for_table = val[table_index][2][fp_errors[label][1][j]-1]

                    if all(e =="-" for e in fp_errors[label][2][j]):
                        message += f"In the table above, Column {fp_errors[label][1][j]} was wrongly classified but the {'correct label is' if len(correct_labels_for_table)<2 else 'correct labels are'} '{', '.join([labels_to_text[l] for l in correct_labels_for_table])}'\n\n"
                    else:
                        if not any(e=="-" for e in fp_errors[label][2][j]):
                            wrong_labels.add(", ".join([labels_to_text[e] for e in fp_errors[label][2][j]]))

                        if label not in fn_errors:
                            message += f"In the table above, Column {fp_errors[label][1][j]} was wrongly classified as '{labels_to_text[label]}' but the {'correct label is' if len(fp_errors[label][2][j])<2 else 'correct labels are'} '{', '.join([labels_to_text[l] for l in fp_errors[label][2][j]])}'\n\n"
                        elif label in fn_errors and set(fp_errors[label][2][j]) - set(correct_labels_for_table) == 0:
                            message += f"In the table above, Column {fp_errors[label][1][j]} was classified as '{', '.join([labels_to_text[l] for l in fp_errors[label][2][j]])}' and is missing the label '{labels_to_text[label]}'\n\n"
                        else:
                            message += f"In the table above, Column {fp_errors[label][1][j]} was wrongly classified as '{', '.join([labels_to_text[l] for l in fp_errors[label][2][j] if l not in correct_labels_for_table and l!='-'])}' but the correct label is '{labels_to_text[label]}'\n\n"

            # 3 random demonstrations of the label to show the model the correct way of using a label
            if label in fn_errors and dataset in ["wikitables-2", "limayeu"]:
                message += "Some examples of the labels that were wrongly used are:\n\n"
                for wrong_label in set([", ".join([labels_to_text[p] for p in pair]) for pair in fn_errors[label][2] if not any(m=="-" for m in pair)]):
                    if wrong_label != "-":
                        try :
                            index = random.choice([j for j, e in enumerate(train_labels) if wrong_label in [",".join(p) for p in parse_json(e).values()]])
                            message += f"{train[index][1]}\n\n"
                            message += f"In the table above, Column {[','.join(p) for p in parse_json(train_labels[index]).values()].index(wrong_label)+1} is an example of label '{wrong_label}'\n\n"
                        except IndexError:
                            continue            
            elif label in fn_errors:
                message += "Some examples of the labels that were wrongly used are:\n\n"
                for wrong_label in set(fn_errors[label][2]):
                    if wrong_label != "-":
                        index = random.choice([j for j, e in enumerate(val) if wrong_label in e[2]])
                        message += f"{val[index][1]}\n\n"
                        message += f"In the table above, Column {val[index][2].index(wrong_label)+1} is an example of label '{labels_to_text[wrong_label]}'\n\n"
                
            pairs = ", ".join([f"'{labels_to_text[label]}' and '{wrong}'" for wrong in wrong_labels])
            human_message = f"For each incorrect label give some tips on how to distinguish them from the label '{labels_to_text[label]}' using the following instructions: 1. Inspect the column values and their associated labels. 2. Based on the above point how could you distinguish the different labels. Give tips only once per pair of labels: {pairs}. If you include examples, don't include whole tables and do not mention specific column numbers. 4. Be concise. Do not give general tips for the label. Start the feedback with tips to distinguish '{labels_to_text[label]}' label."

            messages.append(SystemMessage(content=message))
            messages.append(HumanMessage(content=human_message))

            # pdb.set_trace()

            res = generation_model(messages)
            preds.append(res)
            messages_list.append(messages)
            error_feedback[label] = res.content
        
        # Save comparative definitions
        model_short_name = f"{model_path.split('/')[1].replace('0.0001_16_10_5020_wd=0.0_all-layers_','').replace('0.0001_8_10_5020_wd=0.0_all-layers_','')}"
        save_pickle_file(f"output_val/{dataset}/{model_path}/preds/table-0-shot{model_short_name}-comparative-definitions.pkl", preds)
        save_pickle_file(f"output_val/{dataset}/{model_path}/messages/table-0-shot{model_short_name}-comparative-definitions-messages.pkl", messages_list)
        sienna.save(error_feedback, f"output/{dataset}-{model_short_name}_comparative_definitions.json")
        sienna.save(error_feedback, f"data/{dataset}-labels/{dataset}-{model_short_name}_comparative_definitions.json")
        