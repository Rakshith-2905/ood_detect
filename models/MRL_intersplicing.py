import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


def print_layer_output_sizes(model, input_size):
    hooks = []

    def hook_fn(module, input, output, prefix=''):
        # Print layer's name (with prefix), type, and output size
        print(f"Name:{prefix}\t\tType:{module.__class__.__name__}\t\t Output Size: {output.size()}")

    def register_hooks(module, prefix=''):
        # Register hook for each module and recursively for its children, keeping track of the module hierarchy
        for name, child in module.named_children():
            child_prefix = prefix + '.' + name if prefix else name
            hook = child.register_forward_hook(
                lambda module, input, output, prefix=child_prefix: hook_fn(module, input, output, prefix))
            hooks.append(hook)  # Store the hook for later removal
            register_hooks(child, child_prefix)  # Register hooks recursively for child modules

    # Register hooks and perform a forward pass to print layer outputs
    register_hooks(model)
    dummy_input = torch.randn(input_size)
    model(dummy_input)

    # Remove all registered hooks
    for hook in hooks:
        hook.remove()

class MatryoshkaRemappingNetwork(nn.Module):
    def __init__(self, input_size, layer_specs):
        """
        Initializes the network with a combined specification for channels and spatial dimensions,
        and organizes the remapping and reshaping operations into a sequential model for each segment.

        Parameters:
        - input_size: The size of the input vector.
        - layer_specs: A list of specifications for each layer. Each specification is a tuple, where
          the first element is the number of channels, and the optional second and third elements are
          the spatial dimensions (height, width). If spatial dimensions are not provided, they default to (1, 1).
        """
        super(MatryoshkaRemappingNetwork, self).__init__()

        # Non Linear MLP for initial transformation
        self.mlp = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU()
        )

        self.segments = nn.ModuleList()

        current_size = input_size
        for i, spec in enumerate(layer_specs):
            channel_size = spec[0]
            if len(spec) > 1:
                height, width = spec[1], spec[2]

                output_size = channel_size * height * width
                segment_module = nn.Sequential(
                    nn.Linear(current_size, output_size),
                    nn.Unflatten(1, (channel_size, height, width))  # Reshape to (C, H, W)
                )
            else:
                output_size = channel_size
                segment_module = nn.Sequential(
                    nn.Linear(current_size, output_size),
                )
            self.segments.append(segment_module)
            current_size //= 2

    def forward(self, input_features, use_mlp=False):
        """
        Forward pass through the network, returning a list of remapped and reshaped features for each segment.

        Parameters:
        - input_features: The input vector.
        
        Returns:
        - A list of remapped and reshaped features for each layer specification.
        """
        mapped_features = []

        current_input = input_features
        if use_mlp:
            current_input = self.mlp(input_features)  # Initial transformation

        segment_size = current_input.size(-1)
        for i, _ in enumerate(self.segments):
            segment = current_input[:, :segment_size]  # Get the current segment
            current_input = segment  # Prepare next input
            mapped_feature = self.segments[i](segment)  # Remap and reshape segment

            mapped_features.append(mapped_feature)

            segment_size = current_input.size(-1) // 2  # Update segment size for next iteration

        return mapped_features

class FeatureReplacementWrapper(nn.Module):
    def __init__(self, original_model, replace_layer_names, matryoshka_network, replacement_percentage=50):
        """
        Initializes the feature replacement wrapper with a MatryoshkaRemappingNetwork and a replacement percentage.

        Parameters:
        - original_model: The original model to wrap.
        - replace_layer_names: A list of layer names where feature replacement should occur.
        - matryoshka_network: An instance of MatryoshkaRemappingNetwork to generate replacement features.
        - replacement_percentage: The percentage of channels/neurons to replace with features from the MatryoshkaNetwork. 
                                  Must be between 0 and 100.
        """
        super(FeatureReplacementWrapper, self).__init__()
        self.original_model = original_model
        self.replace_layer_names = replace_layer_names
        self.matryoshka_network = matryoshka_network
        self.replacement_percentage = replacement_percentage
        self._register_hooks()
        self.z_vector = None  # Placeholder for z vector

    def _register_hooks(self):
        """
        Registers forward hooks to perform feature replacement on specified layers.
        """
        for name, module in self.original_model.named_modules():
            if name in self.replace_layer_names:
                module.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        """
        The hook function that replaces the features.
        
        Dynamically replaces a percentage of the channels or neurons with features from the MatryoshkaRemappingNetwork.
        """
        if self.z_vector is None:
            raise ValueError("z vector is not set. Please provide a z vector before forward pass.")

        matryoshka_features = self.matryoshka_features[self.current_hook_index]

        print(f"Output shape: {output.shape}, Matryoshka shape: {matryoshka_features.shape}")

        num_features = output.shape[1]  # Assuming the output shape is [batch_size, channels, H, W]
        num_replace = int(num_features * (self.replacement_percentage / 100.0))  # Calculate how many to replace

        indices_to_replace = torch.randperm(num_features)[:num_replace]

        # Replace the selected features
        modified_output = output.clone()  # Clone to avoid modifying the original output in-place
        for i, idx in enumerate(indices_to_replace):
            
            if len(matryoshka_features.shape) == 4:  # If matryoshka_features include spatial dimensions
                modified_output[:, idx, :, :] = matryoshka_features[:, idx, :, :]
            else:  # If matryoshka_features are flat (no spatial dimensions)
                modified_output[:, idx] = matryoshka_features[:, idx]

        self.current_hook_index += 1  # Increment the hook index for the next layer

        return modified_output
    
    def forward(self, x, z):
        """
        Forward pass through the model. Accepts an input x and a z vector for the Matryoshka network.

        Parameters:
        - x: The input to the original model.
        - z: The z vector to be passed to the Matryoshka network.
        """
        self.z_vector = z  # Set the z vector for use in the hook function

        self.current_hook_index = 0  # Initialize the counter for the hook index
        # Get the remapped features for the z vector from MatryoshkaRemappingNetwork
        self.matryoshka_features = self.matryoshka_network(self.z_vector)  # Use the provided z vector

        result = self.original_model(x)
        self.z_vector = None  # Reset the z vector after use to prevent unintended reuse
        return result


class FeatureReplacementWrapper(nn.Module):
    def __init__(self, original_model, replace_layer_names, matryoshka_network, replacement_percentage=50, loss_layers=[], loss_fn=nn.MSELoss()):
        super(FeatureReplacementWrapper, self).__init__()
        self.original_model = original_model
        self.matryoshka_model = copy.deepcopy(original_model)  # Clone for modifications
        self.replace_layer_names = replace_layer_names
        self.matryoshka_network = matryoshka_network
        self.replacement_percentage = replacement_percentage
        self.loss_layers = set(loss_layers)  # For loss computation
        self.z_vector = None
        self.feature_maps_original = {}  # Features from the original model
        self.feature_maps_matryoshka = {}  # Features from the matryoshka model before replacement
        
        self.loss_fn = loss_fn
        self.current_hook_index = 0  # Counter for the hook index
        self._register_hooks()

    def _register_hooks(self):
        # Iterate through the original model to register capture hooks
        for name, module in self.original_model.named_modules():
            if name in self.loss_layers:
                module.register_forward_hook(self._capture_hook_fn(name, self.feature_maps_original))
        
        # Iterate through the matryoshka model to register both capture and replacement hooks
        for name, module in self.matryoshka_model.named_modules():
            if name in self.loss_layers:
                module.register_forward_hook(self._capture_hook_fn(name, self.feature_maps_matryoshka))
            if name in self.replace_layer_names:
                module.register_forward_hook(self._replacement_hook_fn())

    def _capture_hook_fn(self, layer_name, feature_map_dict):
        def hook(module, input, output):
            # Capture the output features before any potential replacement
            feature_map_dict[layer_name] = output.detach()
        return hook

    def _replacement_hook_fn(self):
        def hook(module, input, output):
            # Perform feature replacement on the output
            return self._replace_features(output)
        return hook

    def _replace_features(self, output):
        if self.z_vector is None:
            raise ValueError("z vector is not set. Please provide a z vector before forward pass.")

        matryoshka_features = self.matryoshka_features[self.current_hook_index]

        print(f"Output shape: {output.shape}, Matryoshka shape: {matryoshka_features.shape}")

        num_features = output.shape[1]  # Assuming the output shape is [batch_size, channels, H, W]
        num_replace = int(num_features * (self.replacement_percentage / 100.0))  # Calculate how many to replace

        indices_to_replace = torch.randperm(num_features)[:num_replace]

        # Replace the selected features
        modified_output = output.clone()  # Clone to avoid modifying the original output in-place
        for i, idx in enumerate(indices_to_replace):
            
            if len(matryoshka_features.shape) == 4:  # If matryoshka_features include spatial dimensions
                modified_output[:, idx, :, :] = matryoshka_features[:, idx, :, :]
            else:  # If matryoshka_features are flat (no spatial dimensions)
                modified_output[:, idx] = matryoshka_features[:, idx]

        self.current_hook_index += 1  # Increment the hook index for the next layer

        return modified_output

    def compute_loss(self):
        loss = 0
        for layer in self.loss_layers:
            original_features = self.feature_maps_original.get(layer)
            matryoshka_features = self.feature_maps_matryoshka.get(layer)
            if original_features is not None and matryoshka_features is not None:
                # If the size of the feature maps is different assert the error
                assert original_features.size() == matryoshka_features.size(), \
                    f"Feature map sizes for layer {layer} do not match: {original_features.size()} != {matryoshka_features.size()}"
                
                print(f"LOSS: Original shape: {original_features.shape}, Matryoshka shape: {matryoshka_features.shape}")
                loss += self.loss_fn(original_features, matryoshka_features)
        return loss

    def forward(self, x, z, compute_loss=True):
        self.z_vector = z
        self.feature_maps_original.clear()
        self.feature_maps_matryoshka.clear()

        self.current_hook_index = 0  # Initialize the counter for the hook index

        self.matryoshka_features = self.matryoshka_network(self.z_vector)  # Use the provided z vector

        # Forward through the original model to capture pre-replacement features
        _ = self.original_model(x)

        # Forward through the matryoshka model for feature replacement and capturing
        result = self.matryoshka_model(x)

        self.z_vector = None  # Reset z_vector after use

        if compute_loss:
            # Compute loss between the captured features
            loss = self.compute_loss()
            return result, loss
        return result

if __name__ == '__main__':
    
    from torchvision.models import resnet18

    model = resnet18()

    # print_layer_output_sizes(model, (1, 3, 224, 224))

    # Define the MatryoshkaRemappingNetwork
    input_size = 512
    mrl_layer_names = ['layer1', 'layer3', 'avgpool']
    loss_layers = ['layer2', 'layer4']

    layer_specs = [
        (64, 56, 56),  # layer1
        (256, 14, 14),  # layer3
        (512,1,1)  # avgpool
    ]

    matryoshka_network = MatryoshkaRemappingNetwork(input_size, layer_specs)
    
    # Wrap the model with the FeatureReplacementWrapper
    wrapped_model = FeatureReplacementWrapper(model, mrl_layer_names, matryoshka_network, replacement_percentage=50, loss_layers=loss_layers, loss_fn=nn.MSELoss())

    # Test the wrapped model
    input_tensor = torch.randn(1, 3, 224, 224)
    z_vector = torch.randn(1, 512)

    z_vector = matryoshka_network.mlp(z_vector)
    output, loss = wrapped_model(input_tensor, z_vector)
    print("Output Size:", output.size())
