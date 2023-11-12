import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101, DeepLabV3_ResNet50_Weights, DeepLabV3_ResNet101_Weights
from torchvision.models.segmentation import fcn_resnet50, fcn_resnet101, FCN_ResNet50_Weights, FCN_ResNet101_Weights
from torchvision import transforms
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor

from PIL import Image
import timm

class CustomFeatureModel(nn.Module):
    def __init__(self, model_name, use_pretrained=False):
        super(CustomFeatureModel, self).__init__()

        self.transform = transforms.Compose([
                            transforms.Resize(224),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])                
                        ])

        supported_models = ['resnet18', 'resnet50', 'resnet101', 'resnet50_adv']
        #  'resnet50x1_bitm', 'resnetv2_101x1_bit.goog_in21k'
        if model_name not in supported_models:
            raise ValueError(f"Invalid model_name. Expected one of {supported_models}, but got {model_name}")

        if model_name == 'resnet50_adv':
            self.model = timm.create_model('resnet50')
            checkpoint = torch.load('./checkpoints/resnet50_l2_eps0.1.ckpt')['model']
            modified_checkpoint = {}
            for k, v in checkpoint.items():
                if 'attacker' in k:
                    continue
                modified_checkpoint[k.replace('module.model.', '')] = v

            self.model.load_state_dict(modified_checkpoint, strict=False) #https://huggingface.co/madrylab/robust-imagenet-models/resolve/main/resnet50_l2_eps0.1.ckpt

            # Remoce the last FC layer
            self.model = nn.Sequential(*list(self.model.children())[:-1])
        else:
            self.model = timm.create_model(model_name, pretrained=use_pretrained, num_classes=0)

        # self.feature_dim = self.model.num_featuress
        self.feature_dim = self.model(torch.zeros(1, 3, 224, 224)).shape[-1]

    def forward(self, x):
            return self.model(x)

class CustomSegmentationModel(nn.Module):
    def __init__(self, model_name, use_pretrained=False):
        super(CustomSegmentationModel, self).__init__()

        supported_models = ['deeplabv3_resnet50', 'deeplabv3_resnet101', 'fcn_resnet50', 'fcn_resnet101']
        if model_name not in supported_models:
            raise ValueError(f"Invalid model_name. Expected one of {supported_models}, but got {model_name}")

        if model_name == 'deeplabv3_resnet50':
            weights = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
        elif model_name == 'deeplabv3_resnet101':
            weights = DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1
        elif model_name == 'fcn_resnet50':
            weights = FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
        elif model_name == 'fcn_resnet101':
            weights = FCN_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1

        self.model = torch.hub.load('pytorch/vision:v0.6.0', model_name, weights=weights)

        self.transform = transforms.Compose([
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])                
                    ])
        self.test_transform = transforms.Compose([
                        transforms.Resize(256, interpolation=3),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])                
                    ])

        # self.transform = transforms.Compose([
        #                     transforms.Resize(520),
        #                     transforms.CenterCrop(520),
        #                     transforms.ToTensor(),
        #                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                         std=[0.229, 0.224, 0.225])                
        #                 ])
                        
        self.feature_model = self.model.backbone
        # Add a max pooling layer with stride 16 to reduce the dimensionality of the features
        self.pool = nn.MaxPool2d(kernel_size=1, stride=1)

        #TODO: add the feature dimension
        

    
    def preprocess_pil(self, images):
        # check if images is a list, then preprocess each image
        if isinstance(images, list):
            images_torch = []
            for image in images:
                image_tensor = self.transform(image)
                images_torch.append(image_tensor)
            images_torch = torch.stack(images_torch).to(device) 
        else:
            images_torch = self.transform(images).unsqueeze(0).to(device)
        
        return images_torch

    @torch.no_grad()
    def forward(self, images):
        if not isinstance(images, torch.Tensor):
            images = self.preprocess_pil(images)
        else:
            if len(images.shape) == 3:
                images = images.unsqueeze(0)
        features = self.feature_model(images)['out']
        features = self.pool(features)
        
        return features

class CustomResNet(nn.Module):
    def __init__(self, model_name, num_classes, use_pretrained=False):
        super(CustomResNet, self).__init__()

        # Define available ResNet architectures
        resnets = {
            'resnet18': resnet18,
            'resnet34': resnet34,
            'resnet50': resnet50,
            'resnet101': resnet101,
            'resnet152': resnet152
        }

        # Check if the provided model_name is valid
        if model_name not in resnets:
            raise ValueError(f"Invalid model_name. Expected one of {list(resnets.keys())}, but got {model_name}")

        # Load the desired ResNet architecture
        self.model = resnets[model_name](pretrained=use_pretrained)

        # Save the features before the FC layer
        self.features = nn.Sequential(*list(self.model.children())[:-1])

        # Update the final fully connected layer to match the number of desired classes
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

        self.feature_dim = self.features(torch.zeros(1, 3, 224, 224)).squeeze(-1).squeeze(-1).shape[-1]

    def forward(self, x, return_features=False):
        if return_features:
            features = self.features(x)
            # Flatten features for the FC layer
            flat_features = features.squeeze(-1).squeeze(-1)

            # Compute logits using the model's FC layer
            logits = self.model.fc(flat_features)

            return logits, flat_features
        else:
            return self.model(x)

# Example usage:
# model = CustomResNet(model_name='resnet50', num_classes=len(train_set.selected_classes))
# model.to(device)


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model = CustomFeatureModel(model_name='resnet50', use_pretrained=True)
    # features = model(torch.zeros(1, 3, 224, 224))
    # print(features.shape)
    # print(model.feature_dim)

    model = CustomSegmentationModel(model_name='deeplabv3_resnet101', use_pretrained=True).to(device)
    pil_image = Image.open("./data/domainnet_v1.0/real/toothpaste/real_318_000284.jpg")
    pil_images = [pil_image]
    torch_tensor = torch.zeros(3, 224, 224).to(device)
    features = model(pil_images)
    print(features.shape)

