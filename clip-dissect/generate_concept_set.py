import requests
import json
import os
from openai import OpenAI
from tqdm import tqdm
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import nltk
from nltk.corpus import stopwords

# Download the list of stop words
nltk.download('stopwords')


# Read the api key from file
with open('api_key.txt', 'r') as f:
    api_key = f.read().strip()
client = OpenAI(
    # This is the default and can be omitted
    api_key=api_key,
)

def get_attributes(class_name, previous_attributes, PROMPT):

    if not PROMPT:
        additional_info = "" if not previous_attributes else f"Exclude these attributes: {', '.join(previous_attributes)}."
        PROMPT = (f"List 100 distinct two-word phrases that describe visual characteristics like(shape, color, texture), "
                f"domain-related attributes, conventional background and general photo elements of a {class_name}."
                f"Make sure the phrases are not long descriptions."
                f"{additional_info}")

    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": PROMPT,
            }
        ],
        model="gpt-3.5-turbo",
    )
    # Extracting the message content
    # message_content = response['choices'][0]['message']['content']

    response = response.choices[0].message.content

    # Parsing the text to create a list without the numbers
    parsed_list = [line.split('. ', 1)[1] for line in response.split('\n') if '. ' in line]

    return parsed_list

def main():
    
    data_name = 'domainnet'
    # Read the prompt template from file json
    with open('prompt_templates.json', 'r') as f:
        prompt_template = json.load(f)[data_name]

    # # Load the classes from the text file
    # with open('data/domainnet_v1.0/class_names.txt', 'r') as f:
    #     classes = f.read().splitlines()  

    classes = ['photo', 'sketch', 'clipart', 'painting', 'infograph', 'quickdraw' ]      
        
    # classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck' ]
    # classes = ['car', 'flower', 'chair', 'truck', 'tiger', 'wheat', 'seal', 'wolf', 'lion', 'dolphin', 'lifeboat', 'corn', 'fishing rod', 'owl', 'sunflower', 'cow', 'bird', 'clock', 'shrimp', 'goose', 'airplane', 'rabbit', 'hot air balloon', 'lizard', 'hat', 'spider', 'motorcycle', 'tortoise', 'dog', 'crocodile', 'elephant', 'gun', 'fox', 'bus', 'cat', 'sailboat', 'giraffe', 'cactus', 'pumpkin', 'train', 'ship', 'helicopter', 'bicycle', 'racket', 'squirrel', 'bear', 'scooter', 'mailbox', 'horse', 'pineapple', 'frog', 'football', 'ostrich', 'tent', 'kangaroo', 'monkey', 'crab', 'sheep', 'butterfly', 'umbrella']
    # classes = ['autumn',  'dim',  'grass',  'outdoor',  'rock',  'water']
    concept_set = {}
    # Iterate over all the classes
    for class_name in tqdm(classes):
        attributes_collected = set()
        for _ in range(5):

            PROMPT = prompt_template.format(class_name=class_name) if not attributes_collected else prompt_template.format(class_name=class_name, additional_info=f"Exclude these descriptions: {', '.join(attributes_collected)}.")
            response = get_attributes(class_name, list(attributes_collected), PROMPT)
            for item in response:
                item_cleaned = item.strip().lower()
                if item_cleaned and item_cleaned not in attributes_collected:
                    attributes_collected.add(item_cleaned)

        concept_set[class_name] = list(attributes_collected)
        
        with open(f'{data_name}_domain_concepts.json', 'w') as f:
            json.dump(concept_set, f, indent=4)

def merge_niccopp():
    with open('NICOpp_att_concepts.json', 'r') as f:
        niccopp_att = json.load(f)
    with open('NICOpp_cls_concepts.json', 'r') as f:
        niccopp_cls = json.load(f)

    nicoopp_concept_set = []
    # Merge all the class concepts values into a single list
    niccopp_cls = list(set([item for sublist in niccopp_cls.values() for item in sublist]))
    
    attributes = ['autumn', 'dim', 'grass', 'outdoor', 'rock', 'water']
    # select these attributes 
    for att in attributes:
        nicoopp_concept_set.extend(niccopp_att[att])
    
    # Load English stop words
    stop_words = set(stopwords.words('english'))

    # Load the 20K concepts from concepts/20k.txt
    with open('concepts/20k.txt', 'r') as f:
        concepts_20k = f.read().splitlines()

    # Remove the stopwords from the 20K concepts
    concepts_20k = [w for w in concepts_20k if not w in stop_words]

    # # add the 20K concepts to the niccopp_concepts
    # nicoopp_concept_set.extend(concepts_20k)

    nicoopp_concept_set = list(set(nicoopp_concept_set))
    # save the concepts as text file
    with open('concepts/niccopp_concepts_att.txt', 'w') as f:
        f.write('\n'.join(nicoopp_concept_set))

if __name__ == "__main__":
    main()
    # merge_niccopp()
