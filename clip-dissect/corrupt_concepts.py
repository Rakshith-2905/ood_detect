import json
import random

def load_json_from_file(file_path):
    """Load JSON data from a specified file."""
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def corrupt_data(data, corruption_percentage):
    """Corrupt the data by mixing values across different keys based on the specified corruption percentage."""
    keys = list(data.keys())
    all_values = {key: data[key][:] for key in keys}  # Create a copy of the lists

    for key in keys:
        n_corrupt = int(corruption_percentage * len(data[key]))
        values_to_replace = random.sample(data[key], n_corrupt)
        
        other_keys = [k for k in keys if k != key]
        replacements = []
        for _ in range(n_corrupt):
            replacement_key = random.choice(other_keys)
            replacement_value = random.choice(all_values[replacement_key])
            replacements.append(replacement_value)
        
        for i, value in enumerate(values_to_replace):
            index = data[key].index(value)
            data[key][index] = replacements[i]

    return data

def save_data_to_file(data, filename):
    """Save the modified data to a specified file."""
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)

# File paths and usage
source_file_path = 'pacs_core_concepts.json'  # Specify the path to your input JSON file
output_file_path = 'pacs_core_concepts_75_corrupted.json'  # Specify the filename to save the corrupted data

# Load data from the source JSON file
data = load_json_from_file(source_file_path)

# Corrupt the data
corruption_percentage = 0.75  # 25% of elements in each list will be corrupted
modified_data = corrupt_data(data, corruption_percentage)

# Save the corrupted data to a file
save_data_to_file(modified_data, output_file_path)
print(f"Data has been successfully corrupted and saved to {output_file_path}")
