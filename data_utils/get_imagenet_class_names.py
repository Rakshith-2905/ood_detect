import json
from imagenet_class_index import IN_CLASS_MAPPING, all_wnids, imagenet_r_wnids, imagenet_a_wnids

class_name_to_idx = {v[-1]: k for k, v in IN_CLASS_MAPPING.items()}
wnids_to_idx = {v[0]: k for k, v in IN_CLASS_MAPPING.items()}
wnids_to_class_names = {v[0]: v[-1] for v in IN_CLASS_MAPPING.values()}
class_names_to_wnids = {v[-1]: v[0] for v in IN_CLASS_MAPPING.values()}

class_names = list(wnids_to_class_names.values())
# replace all "_" with spaces in each of class_names
class_names = [name.replace("_", " ") for name in class_names]
#write this into a json file with each class name in a string quotes and with a comma.
with open('imagenet_class_names.json', 'w') as f:
    json.dump(class_names, f)