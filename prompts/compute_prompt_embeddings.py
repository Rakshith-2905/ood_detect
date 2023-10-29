import torch
import clip
import json

# User-defined paths
CLASS_NAMES_PATH = '../data/domainnet_v1.0/class_names.txt'
TEMPLATES_PATH = 'prompt_templates.json'
OUTPUT_EMBEDDINGS_PATH = 'text_embeddings.pth'
debug = False

# ################# Computing text embeddings #################
# # Load class names from a text file
# with open(CLASS_NAMES_PATH, 'r') as f:
#     class_names = [line.strip() for line in f.readlines()]

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


# if debug == True:
#     # Load the CLIP model and compute embeddings
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model, transform = clip.load("ViT-B/32", device=device)
#     # Load the embeddings
#     all_text_features = torch.load(OUTPUT_EMBEDDINGS_PATH)
#     text_encodings = all_text_features[0]

#     print(len(text_encodings), text_encodings[0].shape)


#     for i, class_name in enumerate(class_names):

#         prompt = f"This is a photo of a {class_name}."

#         text_inputs = clip.tokenize(prompt).to(device)
#         text_features = model.encode_text(text_inputs)

#         # Compute similarities between the text features and the prompt embeddings
#         similarities = text_features @ text_encodings.T

#         j = torch.argmax(similarities).item()

#         # Print the index of the class with the highest similarity in text encodings
#         print(f"Class index: {i}, Class name: {class_names[i]}, Most similar: {class_names[j]}")


################################## Computing image embeddings ##################################
# Load the CLIP model and compute embeddings
device = "cuda" if torch.cuda.is_available() else "cpu"
model, transform = clip.load("ViT-B/32", device=device)

# Load the dataloader
from domainnet_data import DomainNetDataset

domain_train = DomainNetDataset(root_dir='data/domainnet_v1.0', domain="real", split='train')
# Compute the image embeddings
with torch.no_grad():
    all_image_features = []
    for i, (image, label) in enumerate(domain_train):
        print(f"Computing embeddings for image: {i}")
        image_inputs = transform(image).unsqueeze(0).to(device)
        image_features = model.encode_image(image_inputs)
        all_image_features.append(image_features)
        if i == 1500:
            break
image_features_mean = torch.mean(torch.cat(all_image_features, dim=0), dim=0)

# Save the embeddings
torch.save(image_features_mean, "mean_image_embeddings.pth")

all_text_features = torch.load('prompts/text_embeddings.pth')
text_encodings = all_text_features[0]

# Compute mean of text encodings
text_encodings_mean = text_encodings.mean(dim=0)

torch.save(text_encodings_mean, "mean_text_embeddings.pth")
