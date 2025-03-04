from utils import load_cta_dataset, save_pickle_file, load_pickle_file, flatten_list
from evaluation_utils import write_evaluation_file
import pdb

def most_frequent(List):
    return max(set(List), key=List.count)

def all_selected(List):
    all_labels = flatten_list(List)
    set_labels = list(set(flatten_list(List)))
    final_labels = []
    for label in set_labels:
        if label != "-" and all_labels.count(label) > 1:
            final_labels.append(label)
    return final_labels

if __name__ == "__main__":

    dataset = "sotabv2-subsetu"
    # dataset = "limayeu"
    # dataset = "wikitables-2"
    # method = "-hier"
    # method = "-run"
    # method = ""
    method = "-explanation"

    examples, labels, train_examples, train_labels, val_examples, val_labels, labels_to_text, text_to_label, labels_joined, train, val, test = load_cta_dataset(dataset, "-random-20" if dataset!="limayeu" else "") #""
    # pdb.set_trace()
    
    types = list(set([text_to_label[l] for l in text_to_label]))
    
    all_labels = [l for l in labels if l!=""] # change when using wikitables

    model_ids = [
        # "unsloth-Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        # "unsloth-Meta-Llama-3.1-70B-Instruct-bnb-4bit",
        "gpt-4o-2024-05-13",
        # "gpt-4o-mini"
    ]

    for model_id in model_ids:
        all_preds = []
        for file in [
            f"table-0-shot{method}-run_0",
            f"table-0-shot{method}-self-cons-0.5-run_0",
            f"table-0-shot{method}-self-cons-0.7-run_0"
        ]:
            all_preds.append(load_pickle_file(f"output/{dataset}/{model_id}/evaluation/{file}_predictions.pkl"))

        tr_all_preds = list(map(list, zip(*all_preds)))

        voted_preds = []
        if dataset == "sotabv2-subsetu":
            for i in range(len(tr_all_preds)):
                voted_preds.append(most_frequent(tr_all_preds[i]))
        else:
            for i in range(len(tr_all_preds)):
                voted_preds.append(all_selected(tr_all_preds[i]))
        
        # pdb.set_trace()

        save_pickle_file(f"output/{dataset}/{model_id}/evaluation/table-0-shot{method}_self-consistency_predictions.pkl", voted_preds)
        write_evaluation_file(f"output/{dataset}/{model_id}/evaluation/table-0-shot{method}_self-consistency", types, all_labels, voted_preds)