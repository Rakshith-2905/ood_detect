import torch
import torchvision.models as models
import torch.nn as nn
from models.resnet import CustomResNet

import clip

class ResNetFeatures(nn.Module):
    def __init__(self, original_model):
        super(ResNetFeatures, self).__init__()

        original_model = original_model.model
        self.layer1 = nn.Sequential(
            # till layer1
            original_model.conv1,
            original_model.bn1,
            original_model.relu,
            original_model.maxpool,
            original_model.layer1,
        )
        self.layer2 = original_model.layer2
        self.layer3 = original_model.layer3
        self.layer4 = original_model.layer4

        self.last_layer = nn.Sequential(
            original_model.avgpool,
            nn.Flatten(),
            original_model.fc
        )

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        out = self.last_layer(x4)

        return [x1, x2, x3, x4], out

class CLIPResNetFeatures(nn.Module):
    def __init__(self, original_model):
        super(CLIPResNetFeatures, self).__init__()
        
        self.layer1 = nn.Sequential(
            original_model.conv1,
            original_model.bn1,
            original_model.relu1,
            original_model.conv2,
            original_model.bn2,
            original_model.relu2,
            original_model.conv3,
            original_model.bn3,
            original_model.relu3,
            original_model.avgpool,
            original_model.layer1,
        )
        self.layer2 = original_model.layer2
        self.layer3 = original_model.layer3
        self.layer4 = original_model.layer4
        self.final_layer = original_model.attnpool

    def forward(self, x, start_layer="layer1"):
        x = x.type(self.layer1[0].weight.dtype)
        
        if start_layer == "layer1":
            x1 = self.layer1(x)
            x2 = self.layer2(x1)
            x3 = self.layer3(x2)
            x4 = self.layer4(x3)
            x5 = self.final_layer(x4)
            return [x1, x2, x3, x4], x5
        
        elif start_layer == "layer2":
            x2 = self.layer2(x)
            x3 = self.layer3(x2)
            x4 = self.layer4(x3)
            x5 = self.final_layer(x4)
            return [x2, x3, x4], x5
        
        elif start_layer == "layer3":
            x3 = self.layer3(x)
            x4 = self.layer4(x3)
            x5 = self.final_layer(x4)
            return [x3, x4], x5
        
        elif start_layer == "layer4":
            x4 = self.layer4(x)
            x5 = self.final_layer(x4)
            return [x4], x5
        
        else:
            raise ValueError(f"Unknown start_layer: {start_layer}")
       
class ProjectionCNN(nn.Module):
    def __init__(self, input_dim=1024, output_dim=1024, kernel_size=3, padding=1):
        super(ProjectionCNN, self).__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size, padding=padding),  
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.proj(x)


class CombinedModel(nn.Module):
    def __init__(self, clip_model, resnet_model):
        super(CombinedModel, self).__init__()
        
        self.clip_features = CLIPResNetFeatures(clip_model.visual)
        self.resnet_features = ResNetFeatures(resnet_model)
        self.proj_cnn = ProjectionCNN()
        
    def set_trainable_params(self, train_BNLayers=True):
        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False
        
        # Unfreeze only the projection parameters
        for param in self.proj_cnn.parameters():
            param.requires_grad = True

        if train_BNLayers:
            # Unfreeze BatchNorm parameters in the CLIP model
            for module in self.clip_features.modules():
                if isinstance(module, nn.BatchNorm2d):
                    for param in module.parameters():
                        param.requires_grad = True
                        
    def forward(self, x):
        # Extract features using the ResNet model
        (resnet_x1, resnet_x2, resnet_x3, resnet_x4), resnet_out = self.resnet_features(x)
        
        # Project the Layer 3 output of the ResNet model
        projected_resnet_x3 = self.proj_cnn(resnet_x3)
        
        # Extract features using the CLIP model with the projected ResNet features as input for Layer 4
        clip_features, clip_x5 = self.clip_features(projected_resnet_x3, start_layer='layer4')
        
        return clip_x5, resnet_out

if __name__ == "__main__":

    # Load the CLIP model Resnet50
    clip_model, preprocess = clip.load("RN50", device="cuda")
    clip_model.eval()

    resnet_model = CustomResNet('resnet50', num_classes=345 ).to('cuda')

    # Random image
    dummy_image = torch.rand(1, 3, 224, 224).to('cuda')
    dummy_text = ["a photo of a cat", "a photo of a dog"]

    combined_model = CombinedModel(clip_model, resnet_model).to('cuda')
    output = combined_model(dummy_image)
    print(output.shape)

    # Get the CLIP image embeddings
    CLIP_image_embeddings = clip_model.encode_image(dummy_image)
    # Get the CLIP text embeddings
    CLIP_text_embeddings = clip_model.encode_text(clip.tokenize(dummy_text).to('cuda'))

    print(CLIP_image_embeddings.shape, CLIP_text_embeddings.shape)

    custom_features_model = ResNetFeatures(resnet_model).to('cuda')
    features, output = custom_features_model(dummy_image)
    
    CLIP_ResNet_Model = clip_model.visual
    clip_resnet = CLIPResNetFeatures(CLIP_ResNet_Model).to('cuda')    
    clip_features,clip_output = clip_resnet(dummy_image, start_layer='layer1')

    for i in range(len(features)):
        print(f"Layer {i+1}: CLIP  {clip_features[i].shape}\tTask model {features[i].shape}")

    print(f"Final Layer: CLIP  {clip_output.shape}\tTask model {output.shape}")
    


    # print(clip_features[0].shape, features[0].shape)
    # print(clip_features[1].shape, features[1].shape)
    # print(clip_features[2].shape, features[2].shape)
    # print(clip_features[3].shape, features[3].shape)
    # print(clip_features[4].shape, output.shape)

