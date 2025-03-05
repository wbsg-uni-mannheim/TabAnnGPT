import argparse
import os
import torch
from huggingface_hub import login
import logging
from utils import create_save_folders, load_cta_dataset, save_pickle_file, load_pickle_file
from evaluation_utils import evaluate_table, compute_avg_evaluation
import tqdm
from unsloth import FastLanguageModel
import pdb
import sienna

if __name__ == "__main__":

    logger = logging.getLogger(__name__)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(device)

    parser = argparse.ArgumentParser()

    # login(token="HF_token") # Specify HF token if needed
    parser.add_argument("--model_id", default="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit", type=str)
    parser.add_argument("--dataset", type=str, default="sotabv2-subsetu", help="Dataset that is being tested.")
    parser.add_argument("--run", type=str, default="", help="Used to pass definitions or instructions.")
    parser.add_argument("--suff", type=str, default="", help="suffix, used for similar labels or hierarchy info.")
    parser.add_argument("--shot", type=int, default=0, help="Number of similar demonstrations.")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs.")
    parser.add_argument("--run_val", type=bool, default=False, help="Run validation or test set classification.")
    parser.add_argument("--dataset_version", type=str, default="", help="Full training set or random-20 or else.") # in this case where do the similar shots come from
    args = parser.parse_args()
    
    print(args)
    
    output_folder = "output_val" if args.run_val else "output"
    temperature=0.001

    # For self-consistency runs
    if "-self-cons-0.5" in args.run:
        temperature = 0.5
    if "-self-cons-0.7" in args.run:
        temperature = 0.7

    # If testing fine-tuning model
    folder_paths = []
    if "FT" in args.model_id:
        for training_run in os.listdir(f"ft-models/{args.model_id}/"): # where to find the fine-tuned models
            folder_paths.append(f"{args.model_id}/{training_run}") # where to save the predictions
        if args.run !="" and "-self-cons" not in args.run: # For knowledge prompting for fine-tuning scenario, select one of the models
            folder_paths = [folder_paths[0]]
    else:
        folder_paths.append(args.model_id.replace("/","-"))
    

    for folder_path in folder_paths:
        all_eval = []

        for run in range(0,args.runs):
            try:
                if not os.path.exists(f"{output_folder}/{args.dataset}/{folder_path}/preds/table-{args.shot}-shot{args.run}{args.dataset_version}{args.suff}-run_{run}.pkl"):
                    # Load dataset
                    if not args.run_val:
                        examples, labels, train_examples, train_labels, val_examples, val_labels, labels_to_text, text_to_label, labels_joined, train, val, test = load_cta_dataset(args.dataset,args.dataset_version)
                    else:
                        _, _, train_examples, train_labels, examples, labels, labels_to_text, text_to_label, labels_joined, train, test, _ = load_cta_dataset(args.dataset,args.dataset_version)
                    
                    similar_demos = []
                    if args.shot:
                        similar_demos = load_pickle_file(f"embeddings/examples_demonstrations_{args.dataset}{args.dataset_version}_test.pkl")
                    max_seq_length = 5020

                    training_path = f"ft-models/{folder_path}/result/" if "FT" in args.model_id else args.model_id # where to find the (fine-tuned) models
                    torch.cuda.empty_cache()
                    
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
                    if not os.path.exists(f"{output_folder}/{args.dataset}/{folder_path}"):
                        create_save_folders(f"{output_folder}/{args.dataset}/{folder_path}")
                    
                    # Load definitions
                    if args.run !="" and "-self-cons" not in args.run:
                        label_definitions = sienna.load(f"data/{args.dataset}-labels/{args.dataset}{args.run}_definitions.json")
                        # Remove first part of defs so as not to confuse format of labels expected in response if label in format Label: label definition
                        for defn in label_definitions:
                            if "**:" in label_definitions[defn]:
                                split_by_form = label_definitions[defn].split("**:")
                                label_definitions[defn] = "".join(split_by_form[1:]).replace(defn, labels_to_text[defn]).strip()
                        
                        all_labels = [defn for defn in label_definitions]
                        # Show all labels
                        label_definitions_string = "\n".join([ f"{labels_to_text[label]}: {label_definitions[label]}" for label in all_labels if label!="" and label_definitions[label]!="" ])
                        # Show only 10 similar labels
                        if "-similar" in args.suff:
                            if args.run_val:
                                example_labels = load_pickle_file(f"embeddings/examples_labels{args.run}_{args.dataset}_val.pkl")
                            else:
                                example_labels = load_pickle_file(f"embeddings/examples_labels{args.run}_{args.dataset.replace('subsetu','subset')}_test.pkl")
                    
                    messages_list =[]
                    preds = []
                    
                    for j, example in tqdm.tqdm(enumerate(examples), total=len(examples)):
                        messages = []

                        # Task description and instructions
                        instructions = f"You are a world-class data engineer and your task is to annotate the columns of a given table with only one of the following labels that are separated with comma: {labels_joined}."
                        return_format = "{'column_name': 'label'}" if args.dataset not in ["wikitables-2", "limayeu"] else {'column_name': ['label/s']}
                        
                        if args.dataset in ["wikitables-2", "limayeu"]:
                            instructions = f"You are a world-class data engineer and your task is to annotate the columns of a given table with one or more of the following labels that are separated with comma: {labels_joined}."
                            if args.dataset == "wikitables-2" and "hier" in args.suff:
                                hier = sienna.load("data/wikitables-hierarchy-v2.json")
                                instructions += f"\n\nThe labels follow this hierarchy from lowest to highest: {hier}.\n When choosing labels, if the labels are in the lowest hierarchy, choose the higher hierarchies as well."

                        # Add instructions if specified
                        if "-instr" in args.run:
                          instructions += f"\nYour instructions are: 1. Look at the cell values in detail. The first row of the table corresponds to the column names. 2. For each column, select a label that best represents the meaning of all cells in the column. 3. Answer with the selected label for each column using the JSON format {return_format}. 4. Answer only with labels from the provided label set!"
                        
                        # Add definitions if specified
                        if args.run !="" and "-self-cons" not in args.run:
                            if "-similar" in args.suff:
                                label_definitions_string = "\n".join([ f"{labels_to_text[label]}: {label_definitions[label]}" for label in example_labels[j] if label!="" and label_definitions[label]!="" ]) 
                            instructions += f"\nThe definitions of the labels are the following:\n{label_definitions_string}"
                        elif "-instr" not in args.run:
                          instructions += f"\nReply only with the JSON format {return_format}."

                        messages.append({"role": "system", "content": instructions})

                        # Show similar demonstrations
                        if len(similar_demos):
                            for index in similar_demos[j][-args.shot:]:
                                train_extra_message = "Classify these table columns:" 
                                if args.dataset in ["sotabv2-subsetu", "limayeu"]: # asking for particular columns to be annotated
                                    train_extra_message = f"Classify {', '.join([f'Column {col_idx+1}' for col_idx, l in enumerate(train[index][2]) if l!=''])}:"
                                messages.append({"role": "user", "content": f"{train_extra_message}\n{train_examples[index]}"})
                                messages.append({"role": "assistant", "content": f"{train_labels[index]}"})

                        # Add test table
                        extra_message = "Classify these table columns:"
                        if args.dataset in ["sotabv2-subsetu", "limayeu"]:
                            extra_message = f"Classify {', '.join([f'Column {col_idx+1}' for col_idx, l in enumerate(test[j][2]) if l!=''])}:"                      
                        return_message = f"\nReply only with the JSON format {return_format}." if args.run !="" or "-instr" in args.run else ""
                        messages.append({"role": "user", "content": f"{extra_message}\n{example}{return_message}"})

                        messages_list.append(messages)

                        # pdb.set_trace()

                        # Predict column labels
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
                            temperature=temperature,
                            top_p=0.9,
                        )

                        response = outputs[0][input_ids.shape[-1]:]
                        preds.append(tokenizer.decode(response, skip_special_tokens=True))

                    save_pickle_file(f"{output_folder}/{args.dataset}/{folder_path}/preds/table-{args.shot}-shot{args.run}{args.dataset_version}{'-defs' if 'definitions' in args.run else ''}{args.suff}-run_{run}.pkl", preds)
                    save_pickle_file(f"{output_folder}/{args.dataset}/{folder_path}/messages/table-{args.shot}-shot{args.run}{args.dataset_version}{'-defs' if 'definitions' in args.run else ''}{args.suff}-run_{run}-messages.pkl", messages_list)
                    evaluation = evaluate_table(f"{output_folder}/{args.dataset}/{folder_path}/evaluation/table-{args.shot}-shot{args.run}{args.dataset_version}{'-defs' if 'definitions' in args.run else ''}{args.suff}-run_{run}", preds, test, text_to_label)
                    all_eval.append(evaluation)
                else:
                    all_eval.append(sienna.load(f"{output_folder}/{args.dataset}/{folder_path}/evaluation/table-{args.shot}-shot{args.run}{args.dataset_version}{'-defs' if 'definitions' in args.run else ''}{args.suff}-run_{run}_evaluation.json"))
            
            except Exception as error:
                print(folder_path)
                print(f"Error in run number {run}")
                print(error)
                continue

        if len(all_eval) == 3:  
            if f"table-{args.shot}-shot{args.run}{args.dataset_version}{args.suff}-average_evaluation.json" not in os.listdir(f"{output_folder}/{args.dataset}/{folder_path}/evaluation/"):
                # Compute average metrics of the three runs
                compute_avg_evaluation(f"{output_folder}/{args.dataset}/{folder_path}/evaluation/table-{args.shot}-shot{args.run}{args.dataset_version}{'-defs' if 'definitions' in args.run else ''}{args.suff}",all_eval,args.runs)
