import os
import matplotlib.pyplot as plt
import random
random.seed(0)


def plot_class_stats(data_dir='data/domainnet_v1.0'):
    """
    Plot bar plots of image counts for each class in different domains for both train and test sets as subplots.
    Save each plot in the given data directory.
    """
    # Initialize a dictionary to store image counts per class
    class_image_counts = {}

    # List all domains and splits
    domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
    splits = ['train', 'test']

    # Collect the image counts for each domain and split
    for domain in domains:
        class_image_counts[domain] = {}
        for split in splits:
            file_list_path = os.path.join(data_dir, f'text_files/{domain}_{split}.txt')
            if not os.path.isfile(file_list_path):
                print(f"File not found: {file_list_path}")
                continue

            with open(file_list_path, 'r') as file_list:
                for line in file_list:
                    _, class_id = line.strip().split()
                    class_image_counts[domain].setdefault(class_id, {'train': 0, 'test': 0})
                    class_image_counts[domain][class_id][split] += 1

    # Set a clear plot style
    plt.style.use('ggplot')

    # Plotting
    for domain, class_stats in class_image_counts.items():
        # Prepare data for plotting
        classes = sorted(class_stats.keys(), key=int)
        train_counts = [class_stats[cls]['train'] for cls in classes]
        test_counts = [class_stats[cls]['test'] for cls in classes]

        # Create subplots
        fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # Train subplot
        axs[0].bar(classes, train_counts, color='skyblue', label='Train')
        axs[0].set_title(f'{domain.capitalize()} Domain - Training Set')
        axs[0].set_ylabel('Number of Images')
        axs[0].legend()

        # Test subplot
        axs[1].bar(classes, test_counts, color='salmon', label='Test')
        axs[1].set_title(f'{domain.capitalize()} Domain - Test Set')
        axs[1].set_xlabel('Class ID')
        axs[1].set_ylabel('Number of Images')
        axs[1].legend()

        # Improve layout and save the plot
        plt.tight_layout()
        plot_path = os.path.join(data_dir, f"{domain}_class_distribution.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved bar plot for {domain} domain at: {plot_path}")


def select_probe_data(test_file, probe_file):
    # Read the test data
    with open(test_file, 'r') as file:
        lines = file.readlines()

    # Organize data by class
    class_data = {}
    for line in lines:
        path, class_id = line.strip().split()
        if class_id not in class_data:
            class_data[class_id] = []
        class_data[class_id].append(line)

    # Select 10% from each class
    probe_data = []
    for class_id, items in class_data.items():
        ten_percent_count = max(1, len(items) // 10) # Ensure at least one item if class has less than 10 items
        probe_data.extend(random.sample(items, ten_percent_count))

    # Write probe data to a new file
    with open(probe_file, 'w') as file:
        file.writelines(probe_data)

domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
data_dir = 'data/domainnet_v1.0/text_files'
for domain in domains:
    test_file = f"{data_dir}/{domain}_test.txt"
    probe_file = f"{data_dir}/{domain}_probe.txt"
    select_probe_data(test_file, probe_file)

# # Call the function
# plot_class_stats()
