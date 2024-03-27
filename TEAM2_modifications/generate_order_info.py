import json
import openai
from openai import OpenAI
import time



def generate_order_info(dataframe, system_message, few_shot_examples, api_key, model="gpt-4-turbo-preview"):

    """ Generate order information for each transcript in the dataframe using chosen GPT model.
        Requires an OpenAI API key."""

    client = OpenAI(api_key=api_key,)
    sys_message = {"role": "system", "content": system_message}
    start_time = time.time()


    for idx in dataframe.index:

        transcript = dataframe.at[idx, 'Transcript']

        completion = client.chat.completions.create(
            model=model,
            response_format={ "type": "json_object" },
            temperature=0,
            messages=[
            sys_message,
            {"role":"user","content": "Extract the order information for the following transcript:\n"+few_shot_examples['Transcript'].iloc[0]},
            {"role":"assistant","content": json.dumps(few_shot_examples['Order Information'].iloc[0], indent=4)},
            {"role":"user","content": "Extract the order information for the following transcript:\n"+few_shot_examples['Transcript'].iloc[1]},
            {"role":"assistant","content": json.dumps(few_shot_examples['Order Information'].iloc[1], indent=4)},
            {"role":"assistant","content": transcript}
            ]
        )

        new_order_info = json.loads(completion.choices[0].message.content)

        dataframe.at[idx, 'Generated Order Information'] = new_order_info

        if (idx+1) % 25 == 0:
            print(f'Completed {idx+1} samples.')
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f'Elapsed time: {elapsed_time} seconds.')
