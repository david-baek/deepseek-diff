import os
from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI
client = OpenAI()
import json


weight_path = "../checkpoints/version_0/qwen-1.5b_13.pt"
filename = "../results/" + weight_path[2:-3].replace("/","__") + ".json"

with open(filename, 'r') as file:
    data = json.load(file)


# Construct the prompt with the formatted examples.
prompt = (
    "Below are the top 10 examples that most strongly activate a specific sparse autoencoder feature:\n\n"
    "{examples}\n\n"
    "This feature is triggered by the final token and may be influenced by the surrounding context. First, please classify whether this feature activates on math document or general text. Output the feature category in the format: ```Category: <math/general text>```.\n"
    "Then, provide a concise but specific, one-sentence annotation reflecting this classification. If the feature is not clearly interpretable, simply respond with ```Category: uninterpretable```.\n"
    "If the feature seems to be related to reasoning, please provide a type of reasoning in the format: ```Type: <type>```. Below are four types of reasoning features we consider:\n"
    "1. Self-correction: The feature is triggered when the model self-corrects its previous answer, typically by saying 'wait' or 'let me think'.\n"
    "2. Deductive: The feature is triggered when the model uses deductive reasoning to arrive at a conclusion, typically by saying 'therefore' or 'thus'.\n"
    "3. Alternative: The feature is triggered when the model considers alternative solutions or perspectives, for instance, by saying 'alternatively'.\n"
    "4. Contrastive: The feature is triggered when the model compares and contrasts different ideas or concepts.\n"
    "If it does not belong to any of the above types, you do not need to output the type.\n"
    "Note: The feature is activated by the final token, so the signal words will likely be very close to the last token.\n"
)


# Update the system prompt to be more specialized.
system_prompt = (
    "You are an expert in interpreting neural network features. "
    "Carefully analyze the provided examples and deliver a precise, insightful annotation of the feature."
)

base_feat_annotations = dict()

for key in data['base']:
    # Format the top 10 examples into a clear numbered list.
    formatted_examples = "\n".join([f"Example {i+1}. {' '.join(example['context'].split()[-10:])}" for i, example in enumerate(data['base'][key])])
    user_prompt = prompt.format(examples=formatted_examples)
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    # Extract the content from the completion and store it in the dictionary.
    base_feat_annotations[key] = completion.choices[0].message.content
    print(base_feat_annotations[key])

print(base_feat_annotations)

reasoning_feat_annotations = dict()

for key in data['reasoning']:
    # Format the top 10 examples into a clear numbered list.
    formatted_examples = "\n".join([f"Example {i+1}. {' '.join(example['context'].split()[-10:])}" for i, example in enumerate(data['reasoning'][key])])
    user_prompt = prompt.format(examples=formatted_examples)
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    # Extract the content from the completion and store it in the dictionary.
    reasoning_feat_annotations[key] = completion.choices[0].message.content
    print(reasoning_feat_annotations[key])

print(reasoning_feat_annotations)

feat_annotations = {
    "base": base_feat_annotations,
    "reasoning": reasoning_feat_annotations
}

with open(filename[:-5] + "_feat_annotations.json", 'w') as file:
    json.dump(feat_annotations, file, indent=4)