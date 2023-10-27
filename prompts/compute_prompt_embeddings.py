import torch
import clip
import json

# User-defined paths
CLASS_NAMES_PATH = '../data/domainnet_v1.0/class_names.txt'
TEMPLATES_PATH = 'prompt_templates.json'
OUTPUT_EMBEDDINGS_PATH = 'text_embeddings.pth'

# Load class names from a text file
with open(CLASS_NAMES_PATH, 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

# # Load prompt tokens from the JSON file
# with open(TEMPLATES_PATH, 'r') as f:
#     data = json.load(f)

# DEFAULT_TEMPLATE = data["DEFAULT_TEMPLATE"]
# ENSEMBLE_TEMPLATES = data["ENSEMBLE_TEMPLATES"]

# # Combine DEFAULT_TEMPLATE and ENSEMBLE_TEMPLATES for the sake of simplicity
# all_templates = [DEFAULT_TEMPLATE] + ENSEMBLE_TEMPLATES



# # Create combinations
# combinations = [prompt.format(class_name) for class_name in class_names for prompt in all_templates]

# # Load the CLIP model and compute embeddings
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, transform = clip.load("ViT-B/32", device=device)

# with torch.no_grad():
#     all_text_features = []
#     for prompt in all_templates:

#         print(f"Computing embeddings for prompt: {prompt}")

#         prompt_embeddings = []
#         for class_name in class_names:
#             text_inputs = clip.tokenize([prompt.format(class_name)]).to(device)
#             text_features = model.encode_text(text_inputs)
#             prompt_embeddings.append(text_features)
#         prompt_embeddings = torch.cat(prompt_embeddings, dim=0)
#         all_text_features.append(prompt_embeddings)

# # Save the embeddings
# torch.save(all_text_features, OUTPUT_EMBEDDINGS_PATH)

# print(f"Saved text embeddings to {OUTPUT_EMBEDDINGS_PATH}")


# # Load the CLIP model and compute embeddings
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, transform = clip.load("ViT-B/32", device=device)
# # Load the embeddings
# all_text_features = torch.load(OUTPUT_EMBEDDINGS_PATH)
# text_encodings = all_text_features[0]

# print(len(text_encodings), text_encodings[0].shape)


# for i, class_name in enumerate(class_names):

#     prompt = f"This is a photo of a {class_name}."

#     text_inputs = clip.tokenize(prompt).to(device)
#     text_features = model.encode_text(text_inputs)

#     # Compute similarities between the text features and the prompt embeddings
#     similarities = text_features @ text_encodings.T

#     j = torch.argmax(similarities).item()

#     # Print the index of the class with the highest similarity in text encodings
#     print(f"Class index: {i}, Class name: {class_names[i]}, Most similar: {class_names[j]}")