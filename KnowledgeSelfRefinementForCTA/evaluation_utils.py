import pdb
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, hamming_loss, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer
import sienna
from utils import save_pickle_file, parse_json, flatten_list

# Evaluation functions
def decimal(num):
    return np.around(num*100, decimals=2, out=None)

# From:https://mmuratarat.github.io/2020-01-25/multilabel_classification_metrics
def hamming_score(y_true, y_pred):
    temp = 0
    for i in range(y_true.shape[0]):
        temp += sum(np.logical_and(y_true[i], y_pred[i])) / sum(np.logical_or(y_true[i], y_pred[i]))
    return temp / y_true.shape[0]

def hierarchical_f1(labels ,predictions):
    hierarchy_sets = sienna.load("data/wikitables-hierarchy.json")
    intersection_sum = 0
    predicted_set_sum = 0
    label_set_sum = 0

    for i, p in enumerate(predictions):
        
        extended_preds = []
        extended_labels = []
        
        for lab in labels[i]:
            if lab in hierarchy_sets:
                extended_labels += hierarchy_sets[lab]
            else:
                extended_labels += [lab]
        
        extended_labels = list(set(extended_labels))
        
        for lab in p:
            if lab in hierarchy_sets:
                extended_preds += hierarchy_sets[lab]
            else:
                extended_preds += [lab]
        
        extended_preds = list(set(extended_preds))
        
        intersection_sum += len(list(set(extended_preds) & set(extended_labels)))
        predicted_set_sum += len(extended_preds)
        label_set_sum += len(extended_labels)

    hp = intersection_sum / predicted_set_sum
    hr = intersection_sum / label_set_sum

    hf1 = 2*hp*hr / (hp + hr)
    return hf1

# If no OOV answers are predicted, will be similar to sklearn f1 scores, otherwise the score will be lower
def multilabel_calculate_f1_scores(y_tests, y_preds, num_classes, types):
    # Multi-label:
    y_tests = [[types.index(yp) for yp in y ] for y in y_tests]
    y_preds = [[types.index(yp) for yp in y ] for y in y_preds]
    
    #Confusion matrix
    cm = np.zeros(shape=(num_classes,num_classes))
    
    #For multi-label
    for i in range(len(y_tests)):
        for label in y_tests[i]:
            if label in y_preds[i]:
                cm[label][label] += 1
            else:
                cm[-1][label] += 1
        for label in y_preds[i]:
            if label not in y_tests[i]:
                cm[label][-1] += 1
    
    return calculate_f1_p_r(cm, num_classes, types)
   
# If no OOV answers are predicted, will be similar to sklearn f1 scores, otherwise the score will be lower
def calculate_f1_scores(y_tests, y_preds, num_classes, types):

    y_tests = [types.index(y) for y in y_tests]
    y_preds = [types.index(y) for y in y_preds]
    
    #Confusion matrix
    cm = np.zeros(shape=(num_classes,num_classes))
    
    for i in range(len(y_tests)):
        cm[y_preds[i]][y_tests[i]] += 1

    return calculate_f1_p_r(cm, num_classes, types)

def calculate_f1_p_r(cm, num_classes, types):
    report = {}
    
    for j in range(len(cm[0])):
        report[j] = {}
        report[j]['FN'] = 0
        report[j]['FP'] = 0
        report[j]['TP'] = cm[j][j]

        for i in range(len(cm)):
            if i != j:
                report[j]['FN'] += cm[i][j]
        for k in range(len(cm[0])):
            if k != j:
                report[j]['FP'] += cm[j][k]

        if report[j]['TP'] == 0:
            precision = 0
            recall = 0
        else:
            precision = report[j]['TP'] / (report[j]['TP'] + report[j]['FP'])
            recall = report[j]['TP'] / (report[j]['TP'] + report[j]['FN'])

        if np.isnan(precision) or precision == 0:
            f1 = 0
        elif np.isnan(recall) or precision == 0:
            f1 = 0
        else:
            f1 = 2*precision*recall / (precision + recall)

        report[j]['p'] =  precision
        report[j]['r'] =  recall
        report[j]['f1'] = f1
    
    all_fn = 0
    all_tp = 0
    all_fp = 0

    for r in report:
        if r != num_classes-1:
            all_fn += report[r]['FN']
            all_tp += report[r]['TP']
            all_fp += report[r]['FP']
        
    class_f1s = [ report[class_]['f1'] for class_ in report]
    class_p = [ 0 if np.isnan(report[class_]['p']) else report[class_]['p'] for class_ in report]
    class_r = [ 0 if np.isnan(report[class_]['r']) else report[class_]['r'] for class_ in report]
    macro_f1 = sum(class_f1s[:-1]) / (num_classes-1)
    
    p =  sum(class_p[:-1]) / (num_classes-1)
    r =  sum(class_r[:-1]) / (num_classes-1)
    micro_f1 = all_tp / ( all_tp + (1/2 * (all_fp + all_fn) )) 
    
    per_class_eval = {}
    errors_per_class = {}
    for index, t in enumerate(types[:-1]):
        per_class_eval[t] = {"Precision":class_p[index], "Recall": class_r[index], "F1": class_f1s[index]}
        errors_per_class[t] = report[index]["FN"]
    evaluation = {
        "Micro-F1": micro_f1,
        "Macro-F1": macro_f1,
        "Precision": p,
        "Recall": r
    }
    
    return [ evaluation, per_class_eval, errors_per_class]


def map_label(pred, text_to_label):
    if pred in text_to_label:
        return text_to_label[pred]
    else:
        for label in text_to_label:
            if label in pred:
                return text_to_label[label]
            elif pred == text_to_label[label]:
                return text_to_label[label]
    if label == "":
        # In multi-label predictions when the label is empty remove:
        return "emp"
    return "-"


def evaluate_table(path, preds, test, text_to_label):
    
    types = list(set([text_to_label[l] for l in text_to_label]))
    predictions = []
    labels = []
    
    for i, pred in enumerate(preds):
        # Parse JSON string
        pred_j = parse_json(pred)
        if isinstance(pred_j,list):
            print(i)
            pred_j = pred_j[0]
        
        predictions_json = {}
        # Fill the json with correct column names in correct order
        for col_index, label in enumerate(test[i][2]):
            if label != "":
                labels.append(label)
                if isinstance(label, list):
                    predictions_json[test[i][-1][col_index]] = [] # as list for multi-label -> post-processing?
                else:
                    predictions_json[test[i][-1][col_index]] = "-"

        problem = "multi-class" if isinstance(labels[0], str) else "multi-label"
        
        if i == 0:
            print(problem)

        for _, column_name in enumerate(pred_j):
            if column_name in predictions_json:
                try:
                    # Find the column name in the ground truth
                    if problem == "multi-label":
                        if isinstance(pred_j[column_name][0], list): # has explanation
                            for label in pred_j[column_name][0]:
                                predictions_json[column_name].append(map_label(label, text_to_label))
                        else: # no explanation, only labels
                            for label in pred_j[column_name]:
                                predictions_json[column_name].append(map_label(label, text_to_label))
                    elif isinstance(pred_j[column_name], list): # check for explanations
                        predictions_json[column_name] = map_label(pred_j[column_name][0], text_to_label)
                    else:
                        predictions_json[column_name] = map_label(pred_j[column_name], text_to_label)
                except Exception:
                    # If the column name doesn't exist: skip
                    continue
            else:
                for correct_column_name in predictions_json:
                    if column_name == correct_column_name.lower().replace(" ",""):
                        try:
                            # Find the column name in the ground truth
                            if isinstance(pred_j[column_name], list):
                                predictions_json[correct_column_name] = map_label(pred_j[column_name][0], text_to_label)
                            else:
                                predictions_json[correct_column_name] = map_label(pred_j[column_name], text_to_label)
                        except Exception:
                            # If the column name doesn't exist: skip
                            continue
        # pdb.set_trace()
        # Add predictions to list
        for column_name in predictions_json.keys():
            predictions.append(predictions_json[column_name])

    # if multi-label, check that in the predictions there is no duplicate labels
    if problem == "multi-label":
        for j, prediction in enumerate(predictions):
            # remove duplicates
            predictions[j] = list(set(prediction))

            # remove empty answers
            if "emp" in prediction and len(set(prediction))>1:
                predictions[j] = [p for p in predictions[j] if p!="emp"]

    evaluation = write_evaluation_file(path, types, labels, predictions)
    return evaluation
        
def write_evaluation_file(path, types, labels, predictions):
    types = types + ["-"] if '-' in flatten_list(predictions) else types
    if isinstance(labels[0], str):
        lb = LabelBinarizer()
        y_actual = lb.fit_transform(labels)
        y_pred = lb.transform(predictions)
        evaluation, per_class_eval, errors_per_class = calculate_f1_scores(labels, predictions, len(types), types)
        evaluation["sk_micro"] = f1_score(y_actual, y_pred, average="micro")
        evaluation["sk_macro"] = f1_score(y_actual, y_pred, average="macro")
    else:
        mlb = MultiLabelBinarizer() # classes=types
        y_actual = mlb.fit_transform(labels)
        y_pred = mlb.transform(predictions)
        evaluation, per_class_eval, errors_per_class = multilabel_calculate_f1_scores(labels, predictions, len(types), types)
        evaluation["sk_micro"] = f1_score(y_actual, y_pred,average="micro")
        evaluation["sk_macro"] = f1_score(y_actual, y_pred,average="macro")
        evaluation["hamming_loss"] = hamming_loss(y_actual, y_pred)
        evaluation["hamming_score"] = hamming_score(y_actual, y_pred)
        evaluation["exact_match_ratio"] = accuracy_score(y_actual, y_pred)
        if "wikitables" in path:
            evaluation["hf1"] = hierarchical_f1(labels, predictions)

        print(f"HScore: {evaluation['hamming_score']}, EMR: {evaluation['exact_match_ratio']}, HF1: {evaluation['hf1'] if 'wikitables' in path else 'None'}")

    print(f"Precision: {decimal(evaluation['Precision'])}\nRecall: {decimal(evaluation['Recall'])}\nMacro-F1: {decimal(evaluation['Macro-F1'])}\nMicro-F1: {decimal(evaluation['Micro-F1'])}")
    # print("sklearn")
    # sklearn
    # print(f"Micro:{evaluation['sk_micro']}, Macro:{evaluation['sk_macro']}")

    sienna.save(evaluation, f"{path}_evaluation.json")
    sienna.save(per_class_eval, f"{path}_per_class_eval.json")
    save_pickle_file(f"{path}_predictions.pkl", predictions)

    # Error log: label_name, nr_errors, total, % of errors : for the moment works only for multi-class
    if isinstance(labels[0], str):
        error_log = []
        for label in per_class_eval:
            error_log.append([label, errors_per_class[label], len([l for l in labels if l == label]), (errors_per_class[label]/len([l for l in labels if l == label]))])
        pd.DataFrame(error_log, columns=["label","nr_errors","total","perc. errors"]).to_csv(f"{path}_error_log.csv", index=False)

    return evaluation

def compute_avg_evaluation(path, all_eval, runs):
    # Calculate average of the three runs
    avg_eval = {}
    # Fill the dict with metrics available
    for metric in all_eval[0].keys():
        avg_eval[metric] = 0

    for evaluation in all_eval:
        for metric in avg_eval:
            if metric in evaluation:
                avg_eval[metric] += evaluation[metric]
            else:
                print(f"Missing {metric}")
    for metric in avg_eval:
        avg_eval[metric] = avg_eval[metric]/runs

    sienna.save(avg_eval, f"{path}-average_evaluation.json")


def group_errors(val, val_labels, preds):
    # DETECT ERRORS
    fp_errors = {} # incorrect to correct, FPs
    fn_errors = {} # correct to incorrect, FNs
    
    current_nr = 1
    tab_nr = 0
    pred_index = 0
    # COUNT ERRORS
    error_count = {}

    for i, label in enumerate(val_labels):
        if label != "":
            if preds[pred_index] != label and preds[pred_index] !="-":
                if preds[pred_index] not in fp_errors:
                    fp_errors[preds[pred_index]] = [[],[],[]]
                fp_errors[preds[pred_index]][0].append(tab_nr) # table index
                fp_errors[preds[pred_index]][1].append(current_nr) # column index
                fp_errors[preds[pred_index]][2].append(val_labels[i]) # correct labels

                if preds[pred_index] not in error_count:
                    error_count[preds[pred_index]] = 0
                error_count[preds[pred_index]] += 1
            
            pred_index += 1

        if current_nr < len(val[tab_nr][2]):
            current_nr += 1
        elif current_nr == len(val[tab_nr][2]):
            current_nr = 1
            tab_nr += 1

    current_nr = 1
    tab_nr = 0
    pred_index = 0

    for i, label in enumerate(val_labels):
        if label != "":
            if preds[pred_index] != label and preds[pred_index] !="-":
                if label not in fn_errors and label not in fp_errors:
                    fn_errors[label] = [[],[],[]]
                if label not in fp_errors:
                    fn_errors[label][0].append(tab_nr) # table index
                    fn_errors[label][1].append(current_nr) # column index
                    fn_errors[label][2].append(preds[pred_index]) # wrong labels
                if label not in error_count:
                    error_count[preds[pred_index]] = 0
                error_count[preds[pred_index]] += 1
            
            pred_index += 1

        if current_nr < len(val[tab_nr][2]):
            current_nr += 1
        elif current_nr == len(val[tab_nr][2]):
            current_nr = 1
            tab_nr += 1

    for label in fn_errors:
        fp_errors[label] = fn_errors[label]

    return fn_errors, fp_errors


def group_errors_multilabel(val, val_labels, preds):
    # DETECT ERRORS
    fp_errors = {} # incorrect to correct, FPs
    fn_errors = {} # correct to incorrect, FNs
    current_nr = 1
    tab_nr = 0
    pred_index = 0

    for i, labels in enumerate(val_labels):
        if labels != "":
            # check that there is an error
            # pdb.set_trace()
            if not len(set(preds[pred_index]) & set(labels)) == len(labels):
                # FPs
                for label in preds[pred_index]:
                    # pdb.set_trace()
                    if label not in labels and label != "-":
                        # incorrectly predicted
                        if label not in fp_errors:
                            fp_errors[label] = [[],[],[]]
                        
                        fp_errors[label][0].append(tab_nr) # table index
                        fp_errors[label][1].append(current_nr) # column index
                        fp_errors[label][2].append(labels) # correct labels
        
                # FNs
                for label in labels:
                    if label not in preds[pred_index]:
                        # forgotten labels
                        if label not in fn_errors and label not in fp_errors:
                            fn_errors[label] = [[],[],[]]
                        if label not in fp_errors:
                            fn_errors[label][0].append(tab_nr) # table index
                            fn_errors[label][1].append(current_nr) # column index
                            fn_errors[label][2].append(preds[pred_index]) # wrong labels
                        
            pred_index += 1

        if current_nr < len(val[tab_nr][2]):
            current_nr += 1
        elif current_nr == len(val[tab_nr][2]):
            current_nr = 1
            tab_nr += 1
    
    for label in fn_errors:
        fp_errors[label] = fn_errors[label]

    return fn_errors, fp_errors