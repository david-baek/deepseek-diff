import os
from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI
client = OpenAI()
import json


weight_path = "./checkpoints/version_1/qwen-7b_6.pt"
filename = "./results/" + weight_path[2:-3].replace("/","__") + ".json"

with open(filename, 'r') as file:
    data = json.load(file)


# Construct the prompt with the formatted examples.
prompt = (
    "Below are the top 10 examples that most strongly activate a specific sparse autoencoder feature:\n\n"
    "{examples}\n\n"
    "This feature is triggered by the final token and may be influenced by the surrounding context. Given that all texts are math, code, and science reasoning traces, we are seeking a feature that captures a nuance more specific than merely general math and logic. "
    "Please provide a concise, one-sentence annotation that encapsulates the essence of this feature. "
    "If the feature is not interpretable, simply respond with 'uninterpretable.'"
)

# Update the system prompt to be more specialized.
system_prompt = (
    "You are an expert in interpreting neural network features. "
    "Carefully analyze the provided examples and deliver a precise, insightful annotation of the feature."
)

base_feat_annotations = dict()

for key in data['base']:
    # Format the top 10 examples into a clear numbered list.
    formatted_examples = "\n".join([f"Example {i+1}. {example['context']}" for i, example in enumerate(data['base'][key])])
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

print(base_feat_annotations)

reasoning_feat_annotations = dict()

for key in data['reasoning']:
    # Format the top 10 examples into a clear numbered list.
    formatted_examples = "\n".join([f"Example {i+1}. {example['context']}" for i, example in enumerate(data['reasoning'][key])])
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

print(reasoning_feat_annotations)

feat_annotations = {
    "base": base_feat_annotations,
    "reasoning": reasoning_feat_annotations
}

with open(filename[:-5] + "_feat_annotations.json", 'w') as file:
    json.dump(feat_annotations, file, indent=4)