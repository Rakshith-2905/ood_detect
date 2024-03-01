import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

from tqdm import tqdm

from models.mapping import MaxAggregator, MeanAggregator

def kl_divergence_pytorch(p, q):
    """
    Compute the KL divergence between two probability distributions.
    """
    return (p * (p / q).log()).sum(dim=1)

def optimize_attribute_weights(pim_attribute_dict, reference_classes, num_attributes_per_class, learning_rate=0.01, num_optimization_steps=50, aggregation_fn=None):
    """
    Optimize attribute weights for each top-ranked class per instance such that the true class's logit is maximized.
    
    Args:
    :param pim_attribute_dict: A list of dictionaries for each instance in the batch with class indices as keys and tensors of shape [num_attributes] as values.
    :param reference_classes: A tensor of shape [batch_size] with the indices of the reference class for each instance.
    :param num_attributes_per_class: list of number of attributes for each class
    :param learning_rate: The learning rate for the optimizer.
    :param num_optimization_steps: The number of steps for the optimization process.

    Return: A list containing optimized weights for each instance, 
            where each element is a dictionary with class indices as keys and tensors of shape [num_attributes] as values.
    """
    
    batch_size = len(pim_attribute_dict)
    num_classes = len(num_attributes_per_class)
    
    optimized_attribute_weights_batch = []
    # Main loop over the batch
    for i in range(batch_size):
        optimized_attribute_weights = {class_idx: 
                                   torch.ones(num_attributes_per_class[class_idx]) 
                                            for class_idx in range(num_classes)}
        
        # Extract predictions for the current instance
        # dict of class specific attribute logits for the current instance
        instance_predictions = pim_attribute_dict[i]
        
        # Compute aggregated logits by agreegating the attributes for each class using the aggregation function
        # The function should take a dictionary of class specific attribute logits and return a tensor of shape [num_classes]
        aggregated_logits = aggregation_fn(instance_predictions) 

        # Get the ranking of classes based on logits
        _, ranked_classes = aggregated_logits.sort(descending=True) # [num_classes]
        if reference_classes is None:
            # Make the second ranked class the reference class
            reference_class = ranked_classes[1]
        else:
            reference_class = reference_classes[i]
            
        # Find index of the true class
        reference_class_idx = (ranked_classes == reference_class).nonzero(as_tuple=True)[0].item()

        # convert to numpy
        ranked_classes = ranked_classes.cpu().numpy()

        # print(f"\nRanked classes: {ranked_classes}, True class: {reference_class}")

        # Iterate over each class that ranks higher than the true class
        for rank, top_class in enumerate(ranked_classes):
            if rank >= reference_class_idx:  # Skip if not ranked above the true class
                break

            # Initialize weights for optimization for the current top class
            # Set the weights to a large value to support sigmoid
            weights = nn.Parameter(torch.ones(num_attributes_per_class[top_class], requires_grad=True)*3)
            
            # Set up the optimizer for the weights
            optimizer = torch.optim.Adam([weights], lr=learning_rate)

            pbar = tqdm(range(num_optimization_steps), desc=f'Instance {i}')
            # Optimization loop for adjusting weights for the current top class
            for _ in pbar:

                optimizer.zero_grad()
                
                weights_sig = F.sigmoid(weights)  # Apply sigmoid to ensure weights are between 0 and 1
                # Adjust logits for the top class based on current weights
                adjusted_predictions = instance_predictions[top_class] * weights_sig
                # Re-aggregate the logits for the top class
                adjusted_logit = aggregation_fn({top_class: adjusted_predictions})
                
                # Calculate the loss to ensure true class logit is higher than top class logit using leaky relu
                loss = F.leaky_relu(adjusted_logit - aggregated_logits[reference_class_idx], negative_slope=0.1) 

                # Sparse constraint on the weights makes one of the weights close to 0
                # loss_weights = F.relu(weights_sig.min()) + F.relu(1 - weights_sig.max())
                loss_weights = torch.norm(weights_sig, p=1)

                loss += 0.01 * loss_weights


                # Perform gradient descent
                loss.backward()
                optimizer.step()
                
                # Break if the adjusted logit for the top class is less than the true class logit
                if loss.item() < 0:
                    break
                
                pbar.set_postfix({'Loss': loss.item()})

            # Store the optimized weights for the top class
            optimized_attribute_weights[top_class] = weights_sig.detach()
        optimized_attribute_weights_batch.append(optimized_attribute_weights)

    return optimized_attribute_weights_batch

def match_probabilities_to_task_model(pim_attribute_dict, task_model_logits, num_attributes_per_class, learning_rate=0.01, num_optimization_steps=50, aggregation_fn=None):
    """
    Adjust attribute weights for all classes together for each sample such that the attribute-aggregated 
    probabilities of the PIM match the task model's probabilities.
    
    :param pim_attribute_dict: A list of dictionaries for each instance in the batch with class indices as keys and tensors of shape [num_attributes] as values.
    :param task_model_logits: A tensor of shape [batch_size, num_classes] representing the task model's probabilities.
    :param num_attributes_per_class: list of number of attributes for each class
    :param learning_rate: The learning rate for the optimizer.
    :param num_optimization_steps: The number of steps for the optimization process.
    
    :return:
    A list containing optimized weights for each instance, where each element is a list of tensors corresponding to the optimized weights for each class.
    """

    batch_size = task_model_logits.shape[0]
    num_classes = task_model_logits.shape[1]


    optimized_attribute_weights_batch = []

    # Main loop over the batch
    for i in range(batch_size):
        optimized_attribute_weights = {class_idx: 
                                   torch.ones(num_attributes_per_class[class_idx]) 
                                            for class_idx in range(num_classes)}
        # Initialize list of parameter tensors to weight all attributes for each class
        weight_params = [torch.ones(num_attributes_per_class[class_idx], requires_grad=True)*3 for class_idx in range(num_classes)]
        # Initalize the weights for the attributes for each class; Make it a parameter to be optimized
        weight_params = [nn.Parameter(weights) for weights in weight_params]

        # Set up the optimizer for the weights
        optimizer = torch.optim.Adam(weight_params, lr=learning_rate)

        # Extract predictions for the current instance from PIM
        instance_predictions = pim_attribute_dict[i]
        # Optimization loop for adjusting weights for all classes together
        pbar = tqdm(range(num_optimization_steps), desc=f'Instance {i+1}/{batch_size}')
        for _ in pbar:
            optimizer.zero_grad()

            # Initialize a tensor to store adjusted logits for all classes
            adjusted_logits = torch.zeros(num_classes)

            # Apply sigmoid to ensure weights are between 0 and 1
            weight_params_sig = [F.sigmoid(weights) for weights in weight_params]

            # Adjust logits for all classes based on current weights
            for class_idx in range(num_classes):
                # Use only the relevant attributes for each class
                num_attributes = num_attributes_per_class[class_idx]
                
                # Adjust logits for the current class based on current weights
                adjusted_predictions = instance_predictions[class_idx] * weight_params_sig[class_idx]
                # Re-aggregate the logits for the current class
                adjusted_logits[class_idx] = aggregation_fn({class_idx: adjusted_predictions})

            adjusted_probs = F.softmax(adjusted_logits, dim=-1).unsqueeze(0)
            task_model_probs = F.softmax(task_model_logits[i].unsqueeze(0), dim=-1)    

            # Calculate the KL divergence to match the task model's probabilities of this instance
            loss = kl_divergence_pytorch(task_model_probs, adjusted_probs)

            # Perform gradient descent
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix({'Loss': loss.item()})

        # Update the weights for every class
        for i, weights in enumerate(weight_params_sig):
            optimized_attribute_weights[i] = weights.detach()
        
        optimized_attribute_weights_batch.append(optimized_attribute_weights)

    return optimized_attribute_weights_batch

class PIM_Explanations(nn.Module):
    def __init__(self, attribute_names_per_class, num_attributes_per_class, learning_rate=0.01, num_optimization_steps=1000, aggregation_fn=None):
        super().__init__()

        """
        Args:
        attribute_names_per_class: A dictionary with class names as keys and tensors of shape [batch_size, num_attributes] as values.
        num_attributes_per_class: A list of integers with the number of attributes for each class.
        learning_rate: The learning rate for the optimizer.
        num_optimization_steps: The number of steps for the optimization process.
        aggregation_fn: A function that takes a dictionary of class specific attribute logits and returns a tensor of shape [num_classes].

        """       
       
        self.attribute_names_per_class = attribute_names_per_class
        self.num_attributes_per_class = num_attributes_per_class
        self.learning_rate = learning_rate
        self.num_optimization_steps = num_optimization_steps
        self.aggregation_fn = aggregation_fn

        self.class_names = list(attribute_names_per_class.keys())
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self.inverse_normalization = T.Normalize(mean=[-m/s for m, s in zip(self.mean, self.std)], std=[1/s for s in self.std])
    
    def identify_topk_candidates(self, attribute_weights, reference_classes, failed_predicted_logits, k):

        """
        Args:
            attribute_weights: A list of dictionaries (len = num of instances) with class indices as keys and tensors of shape [num_attributes] as values.
            reference_classes: A tensor of shape [batch_size] with the indices of the true classes for each instance.
            failed_predicted_logits: A tensor of shape [batch_size, num_classes] with the predicted logits for each instance.
            k: The number of top-k attributes to identify.
        Returns:
            identified_attributes: A list of dictionaries for each instance in the batch with 
                                    class names as keys and a dictionary with 'attribute'(attribute names)  and 'weights' as values.
        """
        
        # for i in range(len(attribute_weights)):
        #     for class_idx, weights in attribute_weights[i].items():
        #         print(f"Instance {i+1}, Class {class_idx}: {weights}")

        batch_size = reference_classes.shape[0]
        identified_attributes_per_instance = []
        # Iterate over every instance in the batch
        for i in range(batch_size):
            identified_attributes = {}
            # Iterate over the attributes weights for each class
            for class_idx, weights in attribute_weights[i].items():
                # If the weights are all ones, skip the class
                if torch.all(weights == 1):
                    continue
                
                class_name = self.class_names[class_idx]
                identified_attributes[class_name] = {
                    'attributes': self.attribute_names_per_class[class_name],
                    'weights': attribute_weights[i][class_idx]  
                }

            identified_attributes_per_instance.append(identified_attributes)
        
        return identified_attributes_per_instance

    def plot_explanations(self, failed_images, identified_attributes_per_instance,
                        true_class_names, pim_class_names, predicted_class_names, max_description_length=800, 
                        save_path=None, choice="KLD"):
        
        failed_images = self.inverse_normalization(failed_images)
        # Convert to uint8
        failed_images = (failed_images * 255.).type(torch.uint8)
        images_np = failed_images.permute(0, 2, 3, 1).cpu().numpy()
        
        # Number of images
        batch_size = failed_images.shape[0]
        
        # Determine the necessary number of columns for each image based on the length of its description
        num_columns = [2] * batch_size  # Start with 2 columns for each: one for the image, one for text
        for i, attributes in enumerate(identified_attributes_per_instance):
            description_text = 'Classes Flipped: \n'
            for class_name, attr_details in attributes.items():
                description_text += f'\nAttributes Weights of {class_name}:\n'
                for attr, weight in zip(attr_details['attributes'], attr_details['weights']):
                    description_text += f"{attr}: {weight:.2f}\n"
            # Determine if additional columns are needed based on the description length
            extra_columns = int(len(description_text) / max_description_length)
            num_columns[i] += extra_columns

        # Set up the plot with a dynamic number of columns
        max_columns = max(num_columns)  # Find the max number of columns needed
        fig, axes = plt.subplots(nrows=batch_size, ncols=max_columns, figsize=(8 * max_columns, batch_size * 6),
                                gridspec_kw={'width_ratios': [3] + [1] * (max_columns - 1)})

        if batch_size == 1:  # Make sure axes is iterable
            axes = [axes]
        axes = np.array(axes)  # Ensure axes is a NumPy array for easy indexing

        # Iterate through each image
        for i, ax_row in enumerate(axes):
            ax_img = ax_row[0]  # Image column is always the first one
            img = images_np[i]
            ax_img.imshow(img)
            ax_img.axis('off')
            
            # Set the title with class names
            ax_img.set_title(f'True: {true_class_names[i]}, PIM: {pim_class_names[i]}\nTask Model: {predicted_class_names[i]}', fontsize=14)
            
            # Prepare and set the text description for attributes and weights
            identified_attributes = identified_attributes_per_instance[i]
            description_text = f'{choice}: \n'
            for class_name, attributes in identified_attributes.items():
                description_text += f'\nAttributes Weights of {class_name}:\n'
                for attr, weight in zip(attributes['attributes'], attributes['weights']):
                    description_text += f"{attr}: {weight:.2f}\n"
            
            # Splitting text into chunks for each necessary column
            text_chunks = [description_text[j:j + max_description_length] for j in range(0, len(description_text), max_description_length)]
            for j, text_chunk in enumerate(text_chunks):
                anchored_text = AnchoredText(text_chunk, loc="upper left", frameon=False, pad=0.5)
                ax_row[j + 1].add_artist(anchored_text)  # j+1 since the first column is for the image
                ax_row[j + 1].axis('off')  # Turn off axis for the text columns

        plt.tight_layout(pad=3, w_pad=0.5, h_pad=0.5)

        plt.subplots_adjust(left=0.5, right=0.95, top=0.95, bottom=0.05, wspace=0.2, hspace=0.4)
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    def get_explanations(self, images, task_model_logits, pim_logits_dict, true_classes, choice='kld', save_path=None):
        """
        Args:
            images: A torch image [batch_size, C, H, W] with the images for which we want to generate explanations.
            task_model_logits: [batch_size, num_classes]
            pim_logits_dict: A class specific dictionary of tensors of shape [batch_size, num_attributes] with thelogits attribute logits for each class.
            true_classes: A tensor of shape [batch_size] with the indices of the true classes for each instance.
            choice: 'kld' or 'logit_flip' (In KLD, we match the probabilities of the PIM to the task model. In logit_flip, we optimize the attribute weights for each top-ranked class per instance such that the true class's logit is maximized.)
        """

        # Convert pim_logits_dict to a list of dictionaries and get pim predictions
        batch_size = true_classes.shape[0]
        num_classes = len(self.num_attributes_per_class)

        pim_attribute_dict = []
        pim_predictions = []
        for i in range(batch_size):
            pim_attribute_dict.append({class_idx: pim_logits_dict[class_idx][i]
                                           for class_idx in range(num_classes)})
            
            # Aggregate the logits for each class and get the max class
            pim_prediction = torch.argmax(self.aggregation_fn(pim_attribute_dict[i]))
            pim_predictions.append(pim_prediction)
        
        task_model_predictions = torch.argmax(task_model_logits, dim=1)
        pim_predictions = torch.stack(pim_predictions)  # [batch_size, num_classes]

        if choice == 'logit_flip':
            # NOTE: We flip the top prediction of the pim model to understand what attributes are important for the top prediction
            attribute_weights = optimize_attribute_weights(pim_attribute_dict, task_model_predictions, 
                                                            self.num_attributes_per_class, self.learning_rate, 
                                                            self.num_optimization_steps, self.aggregation_fn)
        elif choice == 'kld':
            attribute_weights = match_probabilities_to_task_model(pim_attribute_dict, task_model_logits, 
                                                                    self.num_attributes_per_class, self.learning_rate, 
                                                                    self.num_optimization_steps, self.aggregation_fn)
        else:
            raise ValueError(f"Invalid choice: {choice}. Please choose 'kld' or 'logit_flip'.")
        # Get True class names and failed class names
        true_class_names = [self.class_names[i] for i in true_classes]
        pim_class_names = [self.class_names[i] for i in pim_predictions]
        predicted_class_names = [self.class_names[i] for i in task_model_predictions]

        identified_attributes_per_instance = self.identify_topk_candidates(attribute_weights, pim_predictions, 
                                                                           task_model_logits, k=3)
        
        self.plot_explanations(images, identified_attributes_per_instance, 
                               true_class_names, pim_class_names, predicted_class_names, 
                               save_path=save_path, choice=choice)
        
        return identified_attributes_per_instance

if __name__ == '__main__':
    # Test the code
    # Create a dummy data
    batch_size = 1
    num_classes = 5
    num_attributes_per_class = [5, 6, 7, 8, 9]
    aggregator = MeanAggregator(num_classes=num_classes, num_attributes_per_cls=num_attributes_per_class)

    
    # Create a dummy task model logits
    task_model_logits = torch.rand(batch_size, num_classes)
    pim_logits_dict = {i: torch.rand(batch_size, num_attributes_per_class[i]) for i in range(num_classes)}
    true_classes = torch.randint(0, num_classes, (batch_size,))
    failed_predicted_logits = torch.rand(batch_size, num_classes)



    # Create a dummy attribute names for each class name
    attribute_names_per_class = {
        'class_0': ['class_0_attr_0', 'class_0_attr_1', 'class_0_attr_2', 'class_0_attr_3', 'class_0_attr_4'],
        'class_1': ['class_1_attr_0', 'class_1_attr_1', 'class_1_attr_2', 'class_1_attr_3', 'class_1_attr_4', 'class_1_attr_5'],
        'class_2': ['class_2_attr_0', 'class_2_attr_1', 'class_2_attr_2', 'class_2_attr_3', 'class_2_attr_4', 'class_2_attr_5', 'class_2_attr_6'],
        'class_3': ['class_3_attr_0', 'class_3_attr_1', 'class_3_attr_2', 'class_3_attr_3', 'class_3_attr_4', 'class_3_attr_5', 'class_3_attr_6', 'class_3_attr_7'],
        'class_4': ['class_4_attr_0', 'class_4_attr_1', 'class_4_attr_2', 'class_4_attr_3', 'class_4_attr_4', 'class_4_attr_5', 'class_4_attr_6', 'class_4_attr_7', 'class_4_attr_8']
    }

    # Create a dummy images
    failed_images = torch.rand(batch_size, 3, 224, 224)

    # Create an instance of the PIM_Explanations class
    pim_explanations = PIM_Explanations(attribute_names_per_class, num_attributes_per_class, aggregation_fn=aggregator)

    # Get the explanations
    pim_explanations.get_explanations(failed_images, task_model_logits, pim_logits_dict, 
                                      true_classes, choice='logit_flip', save_path='explanations.png')
