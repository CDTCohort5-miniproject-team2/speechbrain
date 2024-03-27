import re

# Step 1 & 2: Find and replace occurrences for combo-large and combo-regular
def restructure_combos(text):
    text = text.replace('(ITEM combo-large)', '')   # Remove instances of combo which were incorrect entries
    text = text.replace('(ITEM combo-regular)', '') 
    text = text.replace('(ITEM combo-large(', ')(ITEM combo-large)(') # Change nesting of combo-large and combo-regular
    text = text.replace('(ITEM combo-regular(', ')(ITEM combo-regular)(')
    text = text.replace('(ITEM combo-large))', ')(ITEM combo-large)(') 
    text = text.replace('(ITEM combo-regular))', ')(ITEM combo-regular)(')
    return text

# Step 3 & 4: Find and replace occurrences for combo-regular and combo-large ending with ')))'
def remove_redundant_parentheses(text):
    # This regex matches the patterns ending with three ")))" and captures the start until the last ")"
    text = re.sub(r'(\(ITEM combo-large[^\)]+)\)\)\)', r'\1)', text)
    text = re.sub(r'(\(ITEM combo-regular[^\)]+)\)\)\)', r'\1)', text)
    return text

def step_1_reorganise_hierarchy(text):

    text = restructure_combos(text)
    text = remove_redundant_parentheses(text)

    return text

def step_2_split_items(text):
    text = text.strip("()") + ')'

    items = []
    buffer = ""
    nested_level = 0

    for char in text:
        if char == "(":
            nested_level += 1
        elif char == ")" and nested_level > 0:
            nested_level -= 1

        if nested_level == 0 and char == ")":
            items.append(buffer + char)
            buffer = ""
        else:
            buffer += char

    if buffer:  # Add any remaining item
        items.append(buffer)

    return items

def step_3_parse_items_final(text):
    items_json = []
    for item_str in text:
        # Find main item
        main_item_matches = re.search(r'ITEM ([^)(]+)', item_str)
        # Find modifications, ensuring they are different from the main item name
        modifications_matches = re.findall(r'\(ITEM ([^)(]+)\)', item_str)

        if main_item_matches:
            main_item_name = main_item_matches.group(1).strip()
            modifications = [mod for mod in modifications_matches if mod.strip() != main_item_name]
            item = {
                "item": main_item_name,
                "modifications": modifications
            }
            items_json.append(item)
    
    return items_json

def step_4_remove_combo_dip(parsed_items_final):

    for order in parsed_items_final:
        # Check if the item is not 'combo-large' or 'combo-regular'
        if order["item"] not in ["combo-large", "combo-regular"]:
            # Remove 'combo-' from the item name
            order["item"] = order["item"].replace("combo-", "")

        if order['modifications'] not in ["combo-large", "combo-regular"]:
            order['modifications'] = [mod.replace("combo-", "") for mod in order['modifications']]

        order['item'] = order['item'].replace("dip-", "")
        order['modifications'] = [mod.replace("dip-", "") for mod in order['modifications']]

    return parsed_items_final

def step_5_add_size_cleanup(text):

    size_keywords = ['small', 'regular', 'large', 'share box']

    for order in text:

        order['item'] = order['item'].replace('-', ' ')

        item_name = order['item']

        for size in size_keywords:
            # Check if the size keyword is in the item name
            if size in item_name:
                # Add the size to the dictionary
                order['size'] = size
                # Remove the size and any following hyphen from the item name
                # and strip to remove leading/trailing spaces if any
                item_name = item_name.replace(size, '').replace('-', ' ').strip()
                order['item'] = item_name
                break  # Stop looking for other sizes once one is found and processed

        for idx, item in enumerate(order['modifications']):
            order['modifications'][idx] = item.replace('wrap-', '')
            
        for idx, item in enumerate(order['modifications']):
            order['modifications'][idx] = item.replace('-', ' ')

        # for idx, item in enumerate(order['modifications']):
        #     if 'sauce' in item:
        #         order['modifications'][idx] = item.replace(' sauce', '')
            
        if 'modifications' in order and not order['modifications']:

            order.pop('modifications')


    return text

def step_6_order_dict(text):

    order = {"order": text}

    return order

def process_labels(text):
    
    text = step_1_reorganise_hierarchy(text)
    items = step_2_split_items(text)
    parsed_items = step_3_parse_items_final(items)
    parsed_items = step_4_remove_combo_dip(parsed_items)
    parsed_items = step_5_add_size_cleanup(parsed_items)
    parsed_items = step_6_order_dict(parsed_items)

    return parsed_items