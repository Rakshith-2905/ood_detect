import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152

class CustomResNet(nn.Module):
    def __init__(self, model_name, num_classes, use_pretrained=True):
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

    def forward(self, x, return_features=False):
        if return_features:
            return self.features(x).squeeze(-1).squeeze(-1)
        else:
            return self.model(x)

# Example usage:
# model = CustomResNet(model_name='resnet50', num_classes=len(train_set.selected_classes))
# model.to(device)
