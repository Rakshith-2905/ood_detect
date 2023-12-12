import openai
import glob
import os
import json
from tqdm import tqdm

# Initialize the OpenAI API with your secret key
openai.api_key = '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'

def get_descriptors_for_category(category_name):
    # prompt = f"Q: List out the visual features that can describe a {category_name}. Think about its appearance, structure, and surrounding environmental elements."\
    #             f"\nA:The visual features are \n-"
    
    prompt = f"Q: List out short (1-2 words) visual features that can describe a {category_name}. "\
            f"Consider its appearance, structure, and surrounding environmental elements."\
            f"\nA: The visual features are \n-"

    response = openai.Completion.create(
        engine="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=100,
        temperature=0.2
    )

    # Parsing the model's response to obtain the features and convert them to a Python list
    features = [line.strip('- ').strip() for line in response.choices[0].text.split('\n') if line.startswith('-')]
    return features

def save_descriptors_to_file(data, filename="descriptors_cifar10.json"):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)  # The indent parameter makes the JSON easy to read

if __name__ == "__main__":
    # categories = [os.path.basename(path) for path in glob.glob('data/domainnet_v1.0/real/*')]

    categories = [
                "Airplane",
                "Automobile",
                "Bird",
                "Cat",
                "Deer",
                "Dog",
                "Frog",
                "Horse",
                "Ship",
                "Truck"
                ]
    categories = sorted(categories, key=lambda x: x.lower())    
    num_iterations = 10  # Number of unique descriptor sets per category
    all_descriptors = {}

    for category in tqdm(categories, desc="Processing categories"):
        category_descriptors = set()  # Using a set to ensure uniqueness

        for _ in range(num_iterations):
            features = get_descriptors_for_category(category)
            category_descriptors.update(features)  # Merge new features

        all_descriptors[category] = list(category_descriptors)
        save_descriptors_to_file(all_descriptors)  # Save after processing each category

    print("Descriptors saved to file!")
