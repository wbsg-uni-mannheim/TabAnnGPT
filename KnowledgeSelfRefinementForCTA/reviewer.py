import pdb
from utils import create_save_folders, load_pickle_file, parse_json, save_pickle_file, load_cta_dataset, flatten_list
from evaluation_utils import evaluate_table, compute_avg_evaluation
import os
import tqdm
import sienna
from unsloth import FastLanguageModel
import torch
from huggingface_hub import login
import logging
import argparse

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()

    login(token="HF_token") # Specify HF token if necessary
    parser.add_argument("--model_id", default="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit", type=str)
    parser.add_argument("--dataset", type=str, default="sotabv2-subsetu", help="Dataset that is being tested.")
    parser.add_argument("--suff", type=str, default="")
    parser.add_argument("--run", type=str, default="", help="Used to pass definitions or instructions.")
    # Possible methods
    # "-reviewer"
    # "-reviewer-explanation"
    # "-reviewer-errors"
    # "-reviewer-errors-explanation"
    parser.add_argument("--method", type=str, default="-reviewer", help="The zero-shot reviewer setup.")
    parser.add_argument("--dataset_version", type=str, default="", help="Full, random-20 or else.")
    args = parser.parse_args()
    
    print(args)

    # If testing fine-tuning model
    folder_paths = []
    if "FT" in args.model_id:
        for training_run in os.listdir(f"ft-models/{args.model_id}/"): # where to find the fine-tuned models
            folder_paths.append(f"{args.model_id}/{training_run}") # where to save the predictions
    else:
        folder_paths.append(args.model_id.replace("/","-"))

    for folder_path in folder_paths[:1]:
        runs = 3
        all_eval = []

        for run in range(0,runs):
            try:
                if not os.path.exists(f"output/{args.dataset}/{folder_path}/preds/table-0-shot{args.run}{args.dataset_version}{args.method}{args.suff}-run_{run}.pkl"):
                    # Load dataset
                    examples, labels, train_examples, train_labels, val_examples, val_labels, labels_to_text, text_to_label, labels_joined, train, val, test = load_cta_dataset(args.dataset,args.dataset_version)
                    max_seq_length = 5020

                    training_path = f"ft-models/{folder_path}/result/" if "FT" in args.model_id else args.model_id # where to find the (fine-tuned) models
                    
                    # Load model and tokenizer
                    model, tokenizer = FastLanguageModel.from_pretrained(
                        model_name = training_path,
                        max_seq_length = max_seq_length, 
                        dtype = None,
                        load_in_4bit = True,
                        cache_dir="hf_cache/",
                        device_map="auto",
                    )
                    FastLanguageModel.for_inference(model)

                    # If saving directory doesn't exist
                    if not os.path.exists(f"output/{args.dataset}/{folder_path}"):
                        create_save_folders(f"output/{args.dataset}/{folder_path}")
                    
                    # Load definitions
                    if args.run !="" and "error" not in args.method:
                        label_definitions = sienna.load(f"data/{args.dataset}-labels/{args.dataset}{args.run}_definitions.json")
                        # Remove first part of defs so as not to confuse format of labels expected in response
                        for defn in label_definitions:
                            if ":" in label_definitions[defn]:
                                label_definitions[defn] = label_definitions[defn].split(":")[1]
                        
                        all_labels = [defn for defn in label_definitions]
                        # Show all labels
                        label_definitions_string = "\n".join([ f"{labels_to_text[label]}: {label_definitions[label]}" for label in all_labels if label!="" and label_definitions[label]!="" ])
                        # Show only 10 similar labels
                        if "similar" in args.suff:
                            example_labels = load_pickle_file(f"embeddings/examples_labels{args.run}_{args.dataset}_test.pkl")

                    messages_list =[]
                    preds = []
                    pr_suff = f"{'-similar' if args.run !="" else ''}{'-hier' if 'wikitables' in args.dataset else ''}"                    
                    # Load previous predictions: first step of self-correction
                    # Zero-shot setup: the predictions of the zero-shot prompting are loaded
                    # Demonstration definitions setup: the predictions of knowledge prompting with demonstration definitions are loaded
                    # Selective comparative definitions setup: the predictions of the zero-shot prompting predictions are loaded
                    previous_preds = load_pickle_file(f"output/{args.dataset}/{folder_path}/preds/table-0-shot{args.run}{'-explanation' if '-explanation' in args.method else ''}{'-defs' if args.run !='' else ''}{pr_suff}-run_0.pkl")

                    if args.dataset in ["wikitables-2", "limayeu"]:
                        return_format = "{column_name: [[label/s], 'explanation']}" if "explanation" in args.method else "{column_name: [label/s]}"
                    else:
                        nr_labels = ""
                        return_format = "{'column_name': ['label', 'explanation']}" if "explanation" in args.method else "{column_name: label}"
                    
                    # Task description and instructions
                    if args.method == "-reviewer":
                        instructions = f"You are a reviewer model that reviews the column classification done by another model. If a column classification is wrong, return the correct classification for all columns corrected. Respond only with the JSON format: {return_format}."
                    elif args.method == "-reviewer-explanation":
                        instructions = f"You are a reviewer model that reviews the column classification done by another model. If a column classification is wrong, return the correct classification for all columns corrected and your explanation why the label chosen by the other model is correct or not and why the new label chosen fits better. Respond only with the JSON format: {return_format}"
                    elif args.method == "-reviewer-errors":
                        instructions = f"You are a reviewer model that reviews the column classification done by another model. A table and a response from the model will be given to you. Check the table, the response, the label set and some guidelines to distinguish between labels. If a column classification is wrong, return the correct classification for all columns corrected. Respond only with the JSON format: {return_format}"
                    elif args.method == "-reviewer-errors-explanation":
                        instructions = f"You are a reviewer model that reviews the column classification done by another model. A table and a response from the model will be given to you. The response contains the label for each column and an explanation by the model why the label was chosen. Check the table, the response, the label set and some guidelines to distinguish between labels. If a column classification is wrong, return the correct classification for all columns corrected and your explanation why the label chosen by the other model is correct or not and why the new label chosen fits better. Respond only with the JSON format: {return_format}"

                    for j, example in tqdm.tqdm(enumerate(examples), total=len(examples)):
                        messages = []

                        messages.append({"role": "system", "content": instructions})

                        error_string = ""
                        if "error" in args.method:
                            error_feedback = sienna.load(f"output/{args.dataset}{'' if 'FT' in args.model_id else '-'+folder_path}{args.run}-comparative_definitions.json")
                            for label in error_feedback:
                                if args.dataset in ["wikitables-2", "limayeu"]:
                                    labels_with_errors = flatten_list(parse_json(previous_preds[j]).values()) if parse_json(previous_preds[j]) and isinstance(list(parse_json(previous_preds[j]).values())[0][0], str) else [] # else flatten_list(previous_preds[j][0].values()) rregullo me vone
                                    if labels_to_text[label] in labels_with_errors:
                                        error_string += f"{error_feedback[label]}\n\n"
                                else:
                                    if labels_to_text[label] in [pr if not isinstance(pr, list) else pr[0] for pr in parse_json(previous_preds[j]).values()]:
                                        error_string += f"{error_feedback[label]}\n\n"
                            # pdb.set_trace()
                            error_string = error_string.strip()

                        if args.dataset in ["sotabv2-subsetu", "limayeu"]:
                            columns_to_annotate = "The columns that needed to be annotated were " + ', '.join([f'Column {col_idx+1}' for col_idx, l in enumerate(test[j][2]) if l!='']) + ".\n\n"
                        else:
                            columns_to_annotate = ""

                        # Add hierarchical information if available for dataset
                        hier_message = ""
                        if args.dataset in ["wikitables-2"]:
                            hier = sienna.load("data/wikitables-hierarchy-v2.json")
                            hier_message = f"The labels follow this hierarchy from lowest to highest: {hier}\n\n"
                            
                        # Show 10 most similar labels
                        if args.run !="" and "-similar" in args.suff:
                            label_definitions_string = "\n".join([ f"{labels_to_text[label]}: {label_definitions[label]}" for label in example_labels[j] if label!="" and label_definitions[label]!="" ])
                        
                        # Labels set, test table, and previous predictions (+ definitions if included)
                        if args.run !="" and error_string!="": # defs and errors
                            extra_message = f"The label set is: {labels_joined}.\n\n{hier_message}The definitions of the labels are:\n{label_definitions_string}\n\nSome guidelines to distinguish between labels are:\n{error_string}\n\nThe table is:\n{example}\n\n{columns_to_annotate}The response from the classification model is:\n{previous_preds[j]}"
                        if args.run !="" and error_string == "": # defs no errors
                            extra_message = f"The label set is: {labels_joined}.\n\n{hier_message}The definitions of the labels are:\n{label_definitions_string}\n\nThe table is:\n{example}\n\n{columns_to_annotate}The response from the classification model is:\n{previous_preds[j]}"
                        elif error_string !="": # no defs only errors
                            extra_message = f"The label set is: {labels_joined}.\n\n{hier_message}Some guidelines to distinguish between labels are:\n{error_string}\n\nThe table is:\n{example}\n\n{columns_to_annotate}The response from the classification model is:\n{previous_preds[j]}"
                        else: # no defs no errors
                            extra_message = f"The label set is: {labels_joined}.\n\n{hier_message}The table is:\n{example}\n\n{columns_to_annotate}The response from the classification model is:\n{previous_preds[j]}"

                        messages.append({"role": "user", "content": f"{extra_message}\n\nProvide a review for {', '.join([f'Column {col_idx+1}' for col_idx, l in enumerate(test[j][2]) if l!=''])}. Respond only with the JSON format: {return_format}."})

                        # Review and correct
                        input_ids = tokenizer.apply_chat_template(
                            messages,
                            add_generation_prompt=True,
                            return_tensors="pt"
                        ).to(model.device)

                        terminators = [
                            tokenizer.eos_token_id,
                            tokenizer.convert_tokens_to_ids("<|eot_id|>")
                        ]

                        outputs = model.generate(
                            input_ids,
                            max_new_tokens=max_seq_length,
                            eos_token_id=terminators,
                            pad_token_id=tokenizer.eos_token_id,
                            do_sample=True,
                            temperature=0.001,
                            top_p=0.9,
                        )

                        response = outputs[0][input_ids.shape[-1]:]
                        preds.append(tokenizer.decode(response, skip_special_tokens=True))
                        messages_list.append(messages)
                        # pdb.set_trace()

                    save_pickle_file(f"output/{args.dataset}/{folder_path}/preds/table-0-shot{args.run}{args.dataset_version}{args.method}{'-defs' if 'definitions' in args.run else ''}{args.suff}-run_{run}.pkl", preds)
                    save_pickle_file(f"output/{args.dataset}/{folder_path}/messages/table-0-shot{args.run}{args.dataset_version}{args.method}{'-defs' if 'definitions' in args.run else ''}{args.suff}-run_{run}-messages.pkl", messages_list)
                    evaluation = evaluate_table(f"output/{args.dataset}/{folder_path}/evaluation/table-0-shot{args.run}{args.dataset_version}{args.method}{'-defs' if 'definitions' in args.run else ''}{args.suff}-run_{run}", preds, test, text_to_label)
                    all_eval.append(evaluation)

                else:
                    all_eval.append(sienna.load(f"output/{args.dataset}/{folder_path}/evaluation/table-0-shot{args.run}{args.dataset_version}{args.method}{'-defs' if 'definitions' in args.run else ''}{args.suff}-run_{run}_evaluation.json"))

            except Exception as error:
                print(folder_path)
                print(f"Run number {run}")
                print(error)
                continue

        if len(all_eval) == 3: 
            # Compute average metrics of the three runs
            compute_avg_evaluation(f"output/{args.dataset}/{folder_path}/evaluation/table-0-shot{args.run}{args.dataset_version}{args.method}{'-defs' if 'definitions' in args.run else ''}{args.suff}",all_eval,runs)