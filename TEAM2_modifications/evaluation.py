from collections import Counter
import numpy as np
import json


def normalise_item_name(name):
    if type(name) == str:
        return name.replace(" ", "").lower()
    
    else:
        return name

def normalise_order(order):
    if 'order' in order:
        for item in order['order']:
            if 'item' in item:
                item['item'] = normalise_item_name(item['item'])
            if 'flavour' in item:
                item['flavour'] = item['flavour'].lower()
            if 'modifications' in item:
                item['modifications'] = [normalise_item_name(mod) for mod in item['modifications']]
            if 'size' in item:
                item['size'] = normalise_item_name(item['size'])

    return order

def precision_recall(tp, fp, fn):

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0

    f_score = (2*precision*recall) / (precision + recall) if precision + recall > 0 else 0

    return precision, recall, f_score

def precision_recall_item(true_output, generated_output):

    if len(true_output['order']) == 0 and len(generated_output) == 0:

        return 0, 0, 0, 0, 0, 0
    
    else:
    
        norm_true = normalise_order(true_output)
        norm_gen = normalise_order(generated_output)

        true_items = [d['item'] for d in norm_true['order']]
        if 'order' in norm_gen:  # Check if 'order' key exists
            gen_items = [d['item'] for d in norm_gen['order']]
        else:
            gen_items = []  # Default to an empty list if 'order' key is not found

        # Counting occurrences of each item
        true_counts = Counter(true_items)
        gen_counts = Counter(gen_items)

        # Calculating TP, FP, FN
        tp = sum(min(gen_counts[item], true_counts[item]) for item in gen_counts) 
        fp = len(gen_items) - tp
        fn = len(true_items) - tp

        precision, recall, f_score = precision_recall(tp, fp, fn)

        return precision, recall, f_score, tp, fp, fn
    
def extract_values(obj):
    """
    Recursively extract all values from a JSON object (dicts and lists).
    """
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield from extract_values(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from extract_values(item)
    else:
        yield obj

def precision_recall_overall(true_output, generated_output):

    norm_true = normalise_order(true_output)
    norm_gen = normalise_order(generated_output)

    values_t = list(extract_values(norm_true))
    values_g = list(extract_values(norm_gen))
    values_t = Counter(values_t)
    values_g = Counter(values_g)

    tp = sum(min(values_g[item], values_t[item]) for item in values_g) 
    fp = sum(values_g[item] for item in values_g) - tp
    fn = sum(values_t[item] for item in values_t) - tp

    precision, recall, f_score = precision_recall(tp, fp, fn)

    return precision, recall, f_score, tp, fp, fn

def weighted_true_positive(true_output, generated_output):

    weighted_tp = 0
    removed_gen = []
    removed_true = []

    for order in enumerate(generated_output['order']):

        # Initialize the matrix to store the weighting scores
        weighting_matrix = np.zeros((len(generated_output['order']), len(true_output['order'])))

        norm_true = normalise_order(true_output)
        norm_gen = normalise_order(generated_output)

        # Iterate over each gen_item in the generated output
        for gen_idx, gen_item in enumerate(norm_gen['order']):
            # Initialize the sub-item match count
            match_count = 0

            if gen_idx in removed_gen:

                continue
            
            # Iterate over each true_item in the true output
            for true_idx, true_item in enumerate(norm_true['order']):
                # Check if the items match
                match_count = 0

                if true_idx in removed_true:

                    continue

                if gen_item['item'] == true_item['item']:
                    # Check the sub-items
                    match_count = 1
                    if 'modifications' in true_item and 'modifications' in gen_item:
                        true_mods = true_item['modifications']
                        gen_mods = gen_item['modifications']
                        
                        true_mod_counts = Counter(true_mods)
                        gen_mod_counts = Counter(gen_mods)

                        match_count += sum(min(gen_mod_counts[item], true_mod_counts[item]) for item in gen_mod_counts) 

                    if 'size' in true_item and 'size' in gen_item:
                        true_size = true_item['size']
                        gen_size = gen_item['size']
                        if true_size == gen_size:
                            match_count += 1

                    if 'flavour' in true_item and 'flavour' in gen_item:
                        true_size = true_item['flavour']
                        gen_size = gen_item['flavour']
                        if true_size == gen_size:
                            match_count += 1
                    
            
                    # Calculate the incorrect and missing sub-items
                    if 'modifications' in gen_item:         
                        incorrect_items = (len(gen_item)+len(gen_item['modifications'])-1) - match_count
                    else:
                        incorrect_items = len(gen_item) - match_count

                    if 'modifications' in norm_true['order'][true_idx]:
                        missing_items = (len(norm_true['order'][true_idx])+
                                        len(norm_true['order'][true_idx]['modifications'])-1) - match_count
                    else:
                        missing_items = len(norm_true['order'][true_idx]) - match_count

                    # Calculate the weighting score
                    weighting_score = (match_count) / (incorrect_items + missing_items + match_count)
                    
                    # Append the weighting score to the matrix
                    weighting_matrix[gen_idx, true_idx] = weighting_score

        if len(weighting_matrix) == 0:
            break

        index = np.unravel_index(np.argmax(weighting_matrix, axis=None), weighting_matrix.shape)
        max_value = weighting_matrix[index]
        removed_gen.append(index[0])
        removed_true.append(index[1])
        weighted_tp += max_value
        
    return weighted_tp

def precision_recall_weighted(true_output, generated_output):

    tp_weighted = weighted_true_positive(true_output, generated_output)
    _, _, _, tp, fp, fn = precision_recall_item(true_output, generated_output)

    precision = tp_weighted / (tp + fp) if tp + fp > 0 else 0
    recall = tp_weighted / (tp + fn) if tp + fn > 0 else 0

    f_score = (2*precision*recall) / (precision + recall) if precision + recall > 0 else 0

    return precision, recall, f_score, tp_weighted

def safe_json_loads(json_string):
    try:
        # Try to parse the JSON string
        return json.loads(json_string), None
    except json.JSONDecodeError as e:
        # Return None or a custom message, and the error
        return None, e
    
def evaluation(dataframe):

    for col in ['precision_item', 'recall_item', 'f_score_item', 'tp_item', 'fp_item', \
                'fn_item', 'precision_ovr', 'recall_ovr', 'f_score_ovr', 'tp_ovr', 'fp_ovr', \
                    'fn_ovr', 'precision_weighted', 'recall_weighted', 'f_score_weighted', 'tp_weighted']:
        dataframe[col] = None

    for index, row in dataframe.iterrows():

        true_output = row['Order Information']
        generated_output = row['Generated Order Information']

        # Convert the single quotes to double quotes to make it valid JSON
        true_output = true_output.replace("'", '"')
        generated_output = generated_output.replace("'", '"')

        # Convert the JSON string to a Python dictionary
        true_output = json.loads(true_output)

        _, error = safe_json_loads(generated_output)

        if error:
            # Handle the error case, e.g., skip this row, log an error, etc.
            print(f"Error parsing JSON for row {index}: {error}")
            continue  # Skip this iteration
        
        generated_output = json.loads(generated_output)

        precision_item, recall_item, f_score_item, tp_item, fp_item, fn_item = \
            precision_recall_item(true_output, generated_output)

        precision_ovr, recall_ovr, f_score_ovr, tp_ovr, fp_ovr, fn_ovr = \
            precision_recall_overall(true_output, generated_output)
        
        if tp_item == 0:
            precision_weighted = 0
            recall_weighted = 0
            f_score_weighted = 0
            tp_weighted = 0

        else:
            precision_weighted, recall_weighted, f_score_weighted, tp_weighted = \
                precision_recall_weighted(true_output, generated_output)

        precision_recall(tp_weighted, fp_item, fn_item)

        dataframe.at[index, 'precision_item'] = precision_item
        dataframe.at[index, 'recall_item'] = recall_item  
        dataframe.at[index, 'f_score_item'] = f_score_item
        dataframe.at[index, 'tp_item'] = tp_item
        dataframe.at[index, 'fp_item'] = fp_item
        dataframe.at[index, 'fn_item'] = fn_item

        dataframe.at[index, 'precision_ovr'] = precision_ovr
        dataframe.at[index, 'recall_ovr'] = recall_ovr
        dataframe.at[index, 'f_score_ovr'] = f_score_ovr
        dataframe.at[index, 'tp_ovr'] = tp_ovr
        dataframe.at[index, 'fp_ovr'] = fp_ovr
        dataframe.at[index, 'fn_ovr'] = fn_ovr

        dataframe.at[index, 'precision_weighted'] = precision_weighted
        dataframe.at[index, 'recall_weighted'] = recall_weighted
        dataframe.at[index, 'f_score_weighted'] = f_score_weighted
        dataframe.at[index, 'tp_weighted'] = tp_weighted