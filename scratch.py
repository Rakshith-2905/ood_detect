import os
import torch
import clip
feat_dir="tmp_feats1"
feats= [os.path.join(feat_dir, f) for f in os.listdir(feat_dir) if f.endswith(".pt")]
data= torch.load(feats[0])
device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
saved_text_features= data["text_features"][:2].to(device)
captions= data["captions"][:2]

clip_model, preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()
clip_text_features= clip_model.encode_text(clip.tokenize(captions,truncate=True).to(device))
#compare the two saved_text_features and clip_text_features
print(torch.allclose(saved_text_features, clip_text_features))
print(torch.linalg.norm(saved_text_features - clip_text_features, dim=-1, ord=1))


