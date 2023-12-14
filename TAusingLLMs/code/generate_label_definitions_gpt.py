import os
import tqdm
from dotenv import dotenv_values
from langchain.chat_models import ChatOpenAI
import random
from langchain.schema import SystemMessage, HumanMessage
from utils import save_pickle_file, load_cpa_dataset, load_cta_dataset, save_pickle_file, load_cpa_dataset, textada_embeddings

if __name__ == "__main__":
    
    # Load env file with API KEY using full path
    config = dotenv_values("/full/path/to/file/key.env")
    os.environ['OPENAI_API_KEY'] = config["OPENAI_API_KEY"]
    OPENAI_API_KEY = config["OPENAI_API_KEY"]

    model_name = "gpt-3.5-turbo-0301"
    # model_name = "gpt-4-0613"

    chat = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0, model=model_name)

    # CTA prompts
    # System messages
    system_messages_tasks = ["S1", "S2", "S3", "S4", "S5"]
    system_messages_content = {
        "S1": "Your task is to generate knowledge facts about some terms that can help in the task of column type annotation! Reply with only one sentence!",
        "S5": "Your task is to generate knowledge facts about some terms that can help in the task of column type annotation and one example value for the term! Reply with only one sentence!",
        "S2": "Your task is to generate definitions about some terms that can help in the task of column type annotation. Reply with only one sentence! Reply with only one sentence!",
        "S3": "Your task is to generate descriptions about some terms that can help in the task of column type annotation. Reply with only one sentence! Reply with only one sentence!",
        "S4": "Your task is to perform column type annotation, e.g. annotate each column of a table with a label that captures the meaning of the values of the column. As preparation for this task, you are asked to generate definitions of the labels. The definitions should be helpful for distinguishing different labels. Reply with only one sentence!",
    }

    # Instructions
    instruction_messages = ["I1", "I2"]
    instruction_messages_content = {
        "I1": "Your instructions are: 1. Look at the input given to you and make tables out of them. 2. The first row of each table are the column types of each column. 3. Look at their statistical and semantic characteristics of the columns. 4.Generate knowledge for the required term by looking at the whole table. 5. Do not generate specific knowledge for each of the columns. 6.Reply only with knowledge facts not examples.",
        "I2": "Your instructions are: 1. Look at the input given to you and make tables out of them. 2. The first row of each table are the column types of each column. 3. Look at their statistical and semantic characteristics of the columns. 4.Generate a definition of the term by looking at the whole table. 5. Do not generate specific knowledge for each of the columns. 6.Reply only with knowledge facts not examples.",
    }

    # A-prompts
    general_prompts = ["A1", "A2", "A3", "A4", "A5", "A6"]
    general_prompts_content = {
        "A1": "Generate some knowledge facts about the term ",
        "A2": "How can you distinguish if some values are about the term ",
        "A3": "Generate some rules to follow if you need to decide if some values are about the term ",
        "A4": "Generate some instructions to follow if you need to decide if some values are about the term ",
        "A5": "What do you know about the term ",
        "A6": "Generate some instructions to follow if you need to decide if some values may belong to the term ",
    }

    # TB-prompts
    table_prompts = ["TB1", "TB2", "TB3", "TB4", "TB5", "TB6", "TB7"]
    table_prompts_content = {
        "TB1": "Generate some knowledge facts about the term ",
        "TB2": "What characteristics can you learn about the term ",
        "TB3": "Learn about the term ",
        "TB4": "Generate some rules to follow if you need to decide if some values are about the term ",
        "TB5": "Generate some instructions to follow if you need to decide if some values are about the term ",
        "TB6": "What semantic characteristics and statistics can you learn about the term ",
        "TB7": "What value patterns can you learn about the term ",
    }

    for dataset in ["t2dv2-webtables", "sotabv2", "sportstables"]:
        examples, labels, test_table_type_labels, train_examples, train_example_labels, train_table_type_labels, labels_to_text, text_to_label, labels_joined, train, test = load_cta_dataset(dataset,"-for-kg")
        all_labels = [labels_to_text[l] for l in labels_to_text]

        # Run A prompts with combination of S messages:
        for system_mess in system_messages_tasks:
            for g in general_prompts:
                print(f"{system_mess}_{g}_prompt_knowledge")
                if f"{system_mess}_{g}_prompt_knowledge.pkl" not in os.listdir(f"knowledge/{model_name}/{dataset}/"):
                    general_knowledge = []
                    for l in tqdm.tqdm(all_labels, total=len(all_labels)):
                        messages = []
                        messages.append(SystemMessage(content=f"{system_messages_content[system_mess]}"))
                        messages.append(HumanMessage(content=f"{general_prompts_content[g]}{l}?"))
                        res = chat(messages)
                        general_knowledge.append(res.content)

                    save_pickle_file(f"knowledge/{model_name}/{dataset}/{system_mess}_{g}_prompt_knowledge.pkl", general_knowledge)
                    save_pickle_file(f"embeddings/{model_name}/{system_mess}_{g}_knowledge_embeddings_{dataset}.pkl", textada_embeddings(general_knowledge, OPENAI_API_KEY))
                else:
                    print(f"Error in: knowledge/{model_name}/{dataset}/{system_mess}_{g}_prompt_knowledge.pkl")
        
        # Run B prompts with combination of S and I messages:
        for system_mess in system_messages_tasks:
            for instructions in instruction_messages:
                for tab in table_prompts:
                    print(f"{system_mess}_{instructions}_{tab}_prompt_knowledge")
                    if f"{system_mess}_{instructions}_{tab}_prompt_knowledge.pkl" not in os.listdir(f"knowledge/{model_name}/{dataset}/"):
                        general_knowledge = []
                        for l in tqdm.tqdm(all_labels,total=len(all_labels)):
                            messages = []

                            messages.append(SystemMessage(content=f"{system_messages_content[system_mess]}"))
                            messages.append(SystemMessage(content=f"{instruction_messages_content[instructions]}"))
                            message_string = f"""{table_prompts_content[tab]}{l} using the following examples:\n"""

                            for i in range(3):
                                # Randomly select some examples:
                                index = random.choice([j for j, e in enumerate(train_example_labels) if l in e])
                                message_string += f"{train_examples[index]}\n\n"

                            message_string = message_string.strip()
                            messages.append(HumanMessage(content=message_string))
                            print(messages)
                            res = chat(messages)
                            general_knowledge.append(res.content)

                        save_pickle_file(f"knowledge/{model_name}/{dataset}/{system_mess}_{instructions}_{tab}_prompt_knowledge.pkl", general_knowledge)
                        save_pickle_file(f"embeddings/{model_name}/{system_mess}_{instructions}_{tab}_knowledge_embeddings_{dataset}.pkl", textada_embeddings(general_knowledge, OPENAI_API_KEY))

                    else:
                        print(f"Error in: knowledge/{model_name}/{dataset}/{system_mess}_{g}_prompt_knowledge.pkl")

    # CPA prompts
    # System messages
    system_messages_tasks = ["S1", "S2", "S3", "S5"]
    system_messages_content = {
        "S1": "Your task is to generate knowledge facts about some terms that can help in the task of column relationship prediction. Reply with only one sentence!",
        "S5": "Your task is to generate knowledge facts about some terms that can help in the task of column relationship prediction and one example value for the term! Reply with only one sentence!",
        "S2": "Your task is to generate definitions about some terms that can help in the task of column relationship prediction. Reply with only one sentence!",
        "S3": "Your task is to generate descriptions about some terms that can help in the task of column relationship prediction. Reply with only one sentence!",
    }


    for dataset in ["t2dv2-webtables", "sotabv2"]:
        print(dataset)
        examples, labels, test_table_type_labels, train_examples, train_example_labels, train_table_type_labels, labels_to_text, text_to_label, labels_joined, train, test = load_cpa_dataset(dataset, "-for-kg", False) #boolean is for cpa as cta
        all_labels = [labels_to_text[l] for l in labels_to_text]
        
        # Run A prompts with combination of S messages:
        for system_mess in system_messages_tasks:
            for g in general_prompts:
                print(f"cpa-{system_mess}_{g}_prompt_knowledge")
                if f"cpa-{system_mess}_{g}_prompt_knowledge.pkl" not in os.listdir(f"knowledge/{model_name}/{dataset}/"):
                    general_knowledge = []
                    for l in tqdm.tqdm(all_labels, total=len(all_labels)):
                        messages = []
                        messages.append(SystemMessage(content=f"{system_messages_content[system_mess]}"))
                        messages.append(HumanMessage(content=f"{general_prompts_content[g]}{l}?"))

                        res = chat(messages)
                        general_knowledge.append(res.content)

                    save_pickle_file(f"knowledge/{model_name}/{dataset}/cpa-{system_mess}_{g}_prompt_knowledge.pkl", general_knowledge)
                    save_pickle_file(f"embeddings/{model_name}/cpa-{system_mess}_{g}_knowledge_embeddings_{dataset}.pkl", textada_embeddings(general_knowledge, OPENAI_API_KEY))
                else:
                    print(f"Error in: knowledge/{model_name}/{dataset}/cpa-{system_mess}_{g}_prompt_knowledge.pkl")
                    
        # Run B prompts with combination of S and I messages:
        for system_mess in system_messages_tasks:
            for tab in table_prompts:
                print(f"cpa-{system_mess}_I_{tab}_prompt_knowledge")

                if f"cpa-{system_mess}_I_{tab}_prompt_knowledge.pkl" not in os.listdir(f"knowledge/{model_name}/{dataset}/"):
                    general_knowledge = []
                    for l in tqdm.tqdm(all_labels,total=len(all_labels)):
                        messages = []

                        messages.append(SystemMessage(content=f"{system_messages_content[system_mess]}"))
                        messages.append(SystemMessage(content=f"Your instructions are: 1. Look at the input given to you and make tables out of them. 2. The first row of each table are the column relationships of each column with the first column of the table. 3. Look at their statistical and semantic characteristics of the columns. 4.Generate knowledge for the required term by looking at the whole table. 5. Do not generate specific knowledge for each of the columns. 6.Reply only with knowledge facts not examples."))
                        message_string = f"""{table_prompts_content[tab]}{l} using the following examples:\n"""

                        for i in range(3):
                            # Randomly select some examples:
                            index = random.choice([j for j, e in enumerate(train_example_labels) if l in e])
                            message_string += f"{train_examples[index]}\n\n"

                        message_string = message_string.strip()
                        messages.append(HumanMessage(content=message_string))

                        res = chat(messages)
                        general_knowledge.append(res.content)

                    save_pickle_file(f"knowledge/{model_name}/{dataset}/cpa-{system_mess}_I_{tab}_prompt_knowledge.pkl", general_knowledge)
                    save_pickle_file(f"embeddings/{model_name}/cpa-{system_mess}_I_{tab}_knowledge_embeddings_{dataset}.pkl", textada_embeddings(general_knowledge, OPENAI_API_KEY))

                else:
                    print(f"Error in: knowledge/{model_name}/{dataset}/cpa-{system_mess}_I_{tab}_prompt_knowledge.pkl")

