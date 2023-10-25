import json
import nltk
from tqdm import tqdm
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, util

nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

RELEVANT_TERMS = ["color", "shape", "size", "structure", "texture"]

def process_descriptor(desc):
    descriptors = []
    if ":" in desc:
        prefix, suffix = [part.strip() for part in desc.split(":", 1)]
        descriptors.append(suffix)
        
        # If prefix contains relevant terms, add it as a separate descriptor
        if any(term in prefix.lower() for term in RELEVANT_TERMS):
            descriptors.append(prefix)
    else:
        descriptors.append(desc)
    return descriptors

def compute_similarity(desc1, desc2):
    embed1 = model.encode(desc1, convert_to_tensor=True)
    embed2 = model.encode(desc2, convert_to_tensor=True)
    return util.pytorch_cos_sim(embed1, embed2).item()

def remove_redundant_descriptors(descriptors, threshold=0.9):
    unique_descriptors = []
    for desc in descriptors:
        if not any(compute_similarity(desc, unique_desc) > threshold for unique_desc in unique_descriptors):
            unique_descriptors.append(desc)
    return unique_descriptors

def split_comma_separated_descriptors(descriptors):
    expanded_descriptors = []
    for descriptor in descriptors:
        expanded_descriptors.extend([d.strip() for d in descriptor.split(",") if d.strip()])
    return expanded_descriptors

# def is_meaningful(descriptor):
#     tokens = nltk.word_tokenize(descriptor)
#     if not tokens:
#         return False
#     stopword_ratio = sum(1 for word in tokens if word.lower() in STOPWORDS) / len(tokens)
#     return stopword_ratio < 0.7

def clean_all_categories(all_descriptors):
    cleaned_data = {}
    for category in tqdm(all_descriptors.keys(), desc="Processing categories"):
        descriptors = filter(lambda x: x and x.strip() != "", all_descriptors[category])  # Remove None, empty or "" descriptors
        # descriptors = split_comma_separated_descriptors(descriptors)
        
        # processed_descriptors = []
        # for desc in descriptors:
        #     processed_descriptors.extend(process_descriptor(desc))
        
        # meaningful_descriptors = [desc for desc in processed_descriptors if is_meaningful(desc)]
        cleaned_data[category] = remove_redundant_descriptors(descriptors)
        
        # Save the results for this category immediately
        with open("clean_descriptors-2.json", 'w') as f:
            json.dump(cleaned_data, f, indent=4)
    return cleaned_data

def cluster_descriptors(train_classes = 'logs/classifier/resnet_resnet18/train_classes.txt', attributes_dict='descriptors_2.json'):

    # Step 1: Read the class names from train_classes.txt
    with open(train_classes, 'r') as f:
        train_classes = [line.strip() for line in f.readlines()]

    # Step 2: Load the dictionary of attributes from clean_description_2.json
    with open(attributes_dict, 'r') as f:
        attribute_dict = json.load(f)

    # Step 3: Extract the attributes for each class in train classes
    attributes_to_save = []
    for class_name in train_classes:
        if class_name in attribute_dict:
            # Here I assume that attributes for each class in the dictionary are also in a list format
            # If not, adjust accordingly
            attributes = attribute_dict[class_name]
            attributes_to_save.extend(attributes)

    # Step 4: Write the extracted attributes to a new text file
    with open('extracted_attributes.txt', 'w') as f:
        for attribute in attributes_to_save:
            if attribute:
                f.write(attribute + '\n')

if __name__ == "__main__":
    # Load data
    with open("descriptors_2.json", 'r') as f:
        data = json.load(f)
        
    # cleaned_descriptors = clean_all_categories(data)
    cluster_descriptors()

    print("Cleaned descriptors saved to clean_descriptors.json")
