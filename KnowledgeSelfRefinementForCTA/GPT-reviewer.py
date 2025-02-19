import pdb
from utils import load_pickle_file, parse_json, save_pickle_file, load_cta_dataset, flatten_list
from evaluation_utils import evaluate_table
import os
from dotenv import dotenv_values
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import tqdm
import sienna


if __name__ == "__main__":
    # Load env file with API KEY using full path
    config = dotenv_values("/full/path/to/file/key.env")
    os.environ['OPENAI_API_KEY'] = config["OPENAI_API_KEY"]
    OPENAI_API_KEY = config["OPENAI_API_KEY"]

    datasets = ["sotabv2-subsetu"] # Other datasets: "limayeu", "wikitables-2"
    dataset_version = ""
    models = ["gpt-4o-mini"] # gpt-4o-2024-05-13
    # Methods:
    # Zero-shot setup: -reviewer-explanation or -reviewer
    # Selective comparative definitions setup: -reviewer-errors-explanation or -reviewer-errors
    methods = ["-reviewer-explanation"]

    for model_name in models:
        print(model_name)
        chat = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0, model=model_name)
        
        for dataset in datasets:
            print(dataset)
            model_path=model_name
            # Load dataset
            examples, labels, train_examples, train_labels, val_examples, val_labels, labels_to_text, text_to_label, labels_joined, train, val, test = load_cta_dataset(dataset, dataset_version)

            for method in methods:
                # To pair self-correction with definitions, pass the name of the definitions, example: "-gpt-4o-2024-05-13_demonstration"
                # For self-correction in zero-shot setup and in the selective comparative definitions setup leave an empty string: ""
                for def_method in ["-gpt-4o-2024-05-13_demonstration"]:

                    first_messages = []
                    nr_labels = ""

                    if dataset in ["wikitables-2","limayeu"]:
                        return_format = "{column_name: [[label/s], 'explanation']}" if "explanation" in method else "{column_name: [label/s]}"
                        nr_labels = "/s"
                    else:
                        return_format = "{'column_name': ['label', 'explanation']}" if "explanation" in method else "{column_name: label}"
                    
                    if method == "-reviewer": # self-correction, without outputting review
                        first_messages.append(SystemMessage(content=f"You are a reviewer model that reviews the column classification done by another model. A table and a response from the model will be given to you. Check the table, the response and the label set. If a column classification is wrong, return the correct classification for all columns corrected. Respond only with the JSON format: {return_format}"))
                    elif method == "-reviewer-explanation": # self-correction with a review in the output
                        first_messages.append(SystemMessage(content=f"You are a reviewer model that reviews the column classification done by another model. A table and a response from the model will be given to you. The response contains the label{nr_labels} for each column and an explanation by the model why the label{nr_labels} {'were' if nr_labels!='' else 'was'} chosen. Check the table, the response and the label set. If a column classification is wrong, return the correct classification for all columns corrected and your explanation why the label{nr_labels} chosen by the other model is correct or not and why the new label{nr_labels} chosen fit{'' if nr_labels!='' else 's'} better. Respond only with the JSON format: {return_format}"))
                    elif method == "-reviewer-errors": # self-correction paired with selective comparative definitions, without outputting review
                        first_messages.append(SystemMessage(content=f"You are a reviewer model that reviews the column classification done by another model. A table and a response from the model will be given to you. Check the table, the response, the label set and some guidelines to distinguish between labels. If a column classification is wrong, return the correct classification for all columns corrected. Respond only with the JSON format: {return_format}"))
                    elif method == "-reviewer-errors-explanation": # self-correction paired with selective comparative definitions, with review in output
                        first_messages.append(SystemMessage(content=f"You are a reviewer model that reviews the column classification done by another model. A table and a response from the model will be given to you. The response contains the label{nr_labels} for each column and an explanation by the model why the label{nr_labels} {'were' if nr_labels!='' else 'was'} chosen. Check the table, the response, the label set and some guidelines to distinguish between labels. If a column classification is wrong, return the correct classification for all columns corrected and your explanation why the label{nr_labels} chosen by the other model is correct or not and why the new label{nr_labels} chosen fit{'' if nr_labels!='' else 's'} better. Respond only with the JSON format: {return_format}"))

                    # Load previous predictions: first step of self-correction
                    # Zero-shot setup: the predictions of the zero-shot prompting are loaded
                    # Demonstration definitions setup: the predictions of knowledge prompting with demonstration definitions are loaded
                    # Selective comparative definitions setup: the predictions of the zero-shot prompting predictions are loaded
                    previous_preds = load_pickle_file(f"output/{dataset}/{model_path}/preds/table-0-shot{def_method}{'-defs' if def_method != '' else ''}{'-hier' if 'wikitables' in dataset else ''}_0.pkl") #{'-explanation' if '-explanation' in method and 'last' not in method else ''}
                    previous_preds = [p.content for p in previous_preds]
                    
                    messages_list = []
                    preds = []

                    if def_method != "":
                        label_definitions = sienna.load(f"data/{dataset}-labels/{dataset}{def_method}_definitions.json")
                        all_labels = [defn for defn in label_definitions]
                        # Show all label definitions
                        label_definitions_string = "\n".join([ f"{labels_to_text[label]}: {label_definitions[label]}" for label in all_labels if label_definitions[label]!="" ])

                    for j, example in tqdm.tqdm(enumerate(examples), total=len(examples)):
                        messages = []

                        # Add task description message
                        for message in first_messages:
                            messages.append(message)

                        # Selective comparative definitions setup: select comparative definitions to show based on the previous models' predictions 
                        comparative_defs_string = ""
                        if "error" in method:
                            comparative_defs = sienna.load(f"output/{dataset}-{model_name}-comparative_definitions.json")
                            for label in comparative_defs:
                                if dataset in ["wikitables-2","limayeu"]:
                                    labels_with_errors = flatten_list(parse_json(previous_preds[j]).values()) if parse_json(previous_preds[j]) and isinstance(list(parse_json(previous_preds[j]).values())[0][0], str) else [] # else flatten_list(previous_preds[j][0].values()) change
                                    if labels_to_text[label] in labels_with_errors:
                                        comparative_defs_string += f"{comparative_defs[label]}\n\n"
                                else:
                                    if labels_to_text[label] in [pr if not isinstance(pr, list) else pr[0] for pr in parse_json(previous_preds[j]).values()]:
                                        comparative_defs_string += f"{comparative_defs[label]}\n\n"
                            comparative_defs_string = comparative_defs_string.strip()

                        if dataset in ["sotabv2-subsetu", "limayeu"]:
                            columns_to_annotate = "The columns that needed to be annotated were " + ', '.join([f'Column {col_idx+1}' for col_idx, l in enumerate(test[j][2]) if l!='']) + ".\n\n"
                        else:
                            columns_to_annotate = ""

                        # Add hierarchical information if available for dataset
                        nr_labels = ""
                        hier_message = ""
                        if dataset in ["wikitables-2"]:
                            hier = sienna.load("data/wikitables-hierarchy-v2.json")
                            hier_message = f"The labels follow this hierarchy from lowest to highest: {hier}\n\n"
                            nr_labels = "/s"
                        
                        # Labels set, test table, and previous predictions (+ definitions if included)
                        if def_method != "" and comparative_defs_string!="": # Both demonstration and selective comparative defs setup
                            messages.append(HumanMessage(content=f"The label set is: {labels_joined}.\n\n{hier_message}The definitions of the labels are:\n{label_definitions_string}\n\nSome guidelines to distinguish between labels are:\n{comparative_defs_string}\n\nThe table is:\n{example}\n\n{columns_to_annotate}The response from the classification model is:\n{previous_preds[j]}"))
                        elif def_method != "" and comparative_defs_string == "": # initial, demonstration, refined definitions setup
                            messages.append(HumanMessage(content=f"The label set is: {labels_joined}.\n\n{hier_message}The definitions of the labels are:\n{label_definitions_string}\n\nThe table is:\n{example}\n\n{columns_to_annotate}The response from the classification model is:\n{previous_preds[j]}"))
                        elif comparative_defs_string !="": # selective comparative definitions setup
                            messages.append(HumanMessage(content=f"The label set is: {labels_joined}.\n\n{hier_message}Some guidelines to distinguish between labels are:\n{comparative_defs_string}\n\nThe table is:\n{example}\n\n{columns_to_annotate}The response from the classification model is:\n{previous_preds[j]}"))
                        else: # zero-shot setup
                            messages.append(HumanMessage(content=f"The label set is: {labels_joined}.\n\n{hier_message}The table is:\n{example}\n\n{columns_to_annotate}The response from the classification model is:\n{previous_preds[j]}"))
                        

                        if "error" in method and def_method != "":
                            argue_message = " and support your decision by mentioning the part of the label definitions and the guidelines provided to you that helped you reach the decision"
                        elif "error" in method:
                            argue_message = " and support your decision by mentioning the part of the guidelines provided to you that helped you reach the decision"
                        elif def_method != "":
                            argue_message = " and support your decision by mentioning the part of the label definitions provided to you that helped you reach the decision"
                        else:
                            argue_message = ""

                        # Task instructions 
                        if "explanation" in method:
                            instructions = f"""Here are some instructions on how to review and give your explanation for each column:
1. Retrieve the values of the specified column.
2. Look at the column values related to the whole table.
3. Look at the classification of the previous model.
4. Argue or agree with this classification. Return this argument or agreement in the explanation{argue_message}.
5. For each column return its label{nr_labels} and your explanation in the correct format."""
                            messages.append(HumanMessage(content=instructions))
                        else:
                            instructions = f"""Here are some instructions on how to review the classification for each column:
1. Retrieve the values of the specified column.
2. Look at the column values related to the whole table.
3. Look at the classification of the previous model.
4. Argue or agree with this classification.
5. For each column return its label{nr_labels} based on your argument above in the correct format."""
                            messages.append(HumanMessage(content=instructions))

                        # pdb.set_trace()
                        
                        res = chat(messages)
                        preds.append(res)
                        messages_list.append(messages)

                    save_pickle_file(f"output/{dataset}/{model_path}/preds/table-0-shot{def_method}{method}.pkl", preds)
                    save_pickle_file(f"output/{dataset}/{model_path}/messages/table-0-shot{def_method}{method}-messages.pkl", messages_list)
                    evaluate_table(f"output/{dataset}/{model_path}/evaluation/table-0-shot{def_method}{method}", [p.content for p in preds], test, text_to_label)
