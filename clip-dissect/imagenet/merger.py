import json
import os
import re

def sorted_files(directory):
    """ List and sort files by numerical order based on the filename pattern. """
    files = os.listdir(directory)
    # Filter and sort files based on the numerical order in the filename
    files = [f for f in files if re.match(r'imagenet_core_\d+_\d+_concepts\.json', f)]
    files.sort(key=lambda x: int(x.split('_')[2]))  # Sort by the second number in the filename
    return files

def merge_json_files(directory, output_filename):
    merged_data = {}
    for filename in sorted_files(directory):
        with open(os.path.join(directory, filename), 'r') as file:
            data = json.load(file)
            merged_data.update(data)  # Update the merged dictionary with the new data

    with open(output_filename, 'w') as outfile:
        json.dump(merged_data, outfile, indent=4)

# Usage example
merge_json_files('./', 'merged_concepts.json')
