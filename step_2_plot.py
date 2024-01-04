import pandas as pd
import matplotlib.pyplot as plt
import os
import math

# Define your lists
teacher_temps = [0.5, 2.0]
weight_img_loss = [0.5, 1.0]
weight_txt_loss = [0.5, 1.0, 2.0]


base_path = "./logs_2/cifar10/all/simple_cnn/plumber_img_text_proj"

# Function to generate file path
def generate_file_path(teT, img_weight, txt_weight):
    file_path = f"{base_path}/_clsEpoch_29_bs_256_lr_0.1_teT_{teT}_sT_1.0_imgweight_{img_weight}_txtweight_{txt_weight}_is_mlp_False/step_2/lightning_logs/version_1/metrics.csv"
    return file_path

def plot_epoch_accuracies(teT):
    plt.figure(figsize=(12, 6))
    base_acc_plotted = False

    for img_weight in weight_img_loss:
        for txt_weight in weight_txt_loss:
            file_path = generate_file_path(teT, img_weight, txt_weight)
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                plt.plot(df['val_plumber_acc'], label=f'img:{img_weight}, txt:{txt_weight}')

                # Plot the base_acc only once and make it stand out
                if not base_acc_plotted:
                    plt.plot(df['val_base_acc'], label='Base Acc (Reference)', color='black', linewidth=2.5, linestyle='--')
                    base_acc_plotted = True

    plt.title(f'Epoch-wise Accuracies for Teacher Temp {teT}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='#cccccc', zorder=0)

    # Place the legend on the plot
    plt.legend(fontsize='small')

    plt.savefig(f"{base_path}/plots/step2/val_acc_teachertemp_{teT}.png")
    plt.clf()  # Clear the figure after saving

def plot_last_epoch_accuracies():
    num_subplots = len(teacher_temps)
    fig, axes = plt.subplots(1, num_subplots, figsize=(20, 6), sharey=True)
    base_acc_plotted = False
    color_palette = plt.cm.get_cmap('tab20', len(weight_img_loss) * len(weight_txt_loss))  # Muted color palette

    for idx, teT in enumerate(teacher_temps):
        acc_values = []
        labels = []
        base_acc = None

        for img_weight_idx, img_weight in enumerate(weight_img_loss):
            for txt_weight_idx, txt_weight in enumerate(weight_txt_loss):
                file_path = generate_file_path(teT, img_weight, txt_weight)
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    last_epoch_acc = df['val_plumber_acc'].iloc[-1]
                    acc_values.append(last_epoch_acc)
                    labels.append(f'img:{img_weight}, txt:{txt_weight}')

                    if base_acc is None:
                        base_acc = df['val_base_acc'].iloc[-1]

                # Plotting the bar chart with higher zorder
                axes[idx].bar(labels[-1], acc_values[-1], color=color_palette(img_weight_idx * len(weight_txt_loss) + txt_weight_idx), zorder=3)

        # Plot the base_acc only once and make it stand out
        if base_acc is not None:
            base_acc_plotted = True
            axes[idx].axhline(y=base_acc, color='black', linewidth=2.5, linestyle='--', label='Base Acc (Reference)' if idx == 0 else "")

        axes[idx].set_title(f'Teacher Temp: {teT}')
        axes[idx].set_xticklabels(labels, rotation=45, ha="right")
        axes[idx].set_ylabel('Accuracy')
        # Adding grid with lower zorder
        axes[idx].grid(True, which='both', linestyle='--', linewidth=0.5, color='#cccccc', zorder=2)

    # Adjusting layout and saving the plot
    if base_acc_plotted:
        # Adjust the position of the legend
        fig.legend()
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust the rectangle in which to fit the subplots
    os.makedirs(f"{base_path}/plots/step2", exist_ok=True)
    plot_dir = f"{base_path}/plots/step2/last_epoch_acc_comparison.png"
    plt.savefig(plot_dir)
    plt.clf()

def plot_last_epoch_accuracies_for_corruptions():
    base_path = "./logs_2/cifar10/all/simple_cnn/plumber_img_text_proj/_clsEpoch_29_bs_256_lr_0.1_teT_0.5_sT_1.0_imgweight_1.0_txtweight_1.0_is_mlp_False/step_2"
    # Define the list of CIFAR-10-C corruptions
    corruptions = [
        "brightness", "gaussian_blur", "gaussian_noise",
        "impulse_noise", "jpeg_compression", "motion_blur",
        # "pixelate"
    ]
    severity_levels = [2, 3, 4]

    # Determine the layout of the subplots
    num_corruptions = len(corruptions)
    num_columns = 2  # Can be adjusted based on preference
    num_rows = math.ceil(num_corruptions / num_columns)
    fig_width = num_columns * 12  # 12 is an arbitrary width for each subplot column
    fig_height = num_rows * 6   # 6 is an arbitrary height for each subplot row

    fig, axes = plt.subplots(num_rows, num_columns, figsize=(fig_width, fig_height))
    axes = axes.flatten() if num_corruptions > 1 else [axes]

    for idx, corruption in enumerate(corruptions):
        plumber_accuracies = []
        base_accuracies = []

        for severity in severity_levels:
            file_path = f"{base_path}/{corruption}-{severity}/lightning_logs/version_1/metrics.csv"
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                last_epoch_plumber_acc = df['val_plumber_acc'].iloc[-1]
                last_epoch_base_acc = df['val_base_acc'].iloc[-1]
                plumber_accuracies.append(last_epoch_plumber_acc)
                base_accuracies.append(last_epoch_base_acc)
            else:
                plumber_accuracies.append(None)
                base_accuracies.append(None)

        axes[idx].plot(severity_levels, plumber_accuracies, marker='o', label='Plumber Acc')
        axes[idx].plot(severity_levels, base_accuracies, marker='x', label='Base Acc')
        axes[idx].set_title(corruption)
        axes[idx].set_xlabel('Severity Level')
        axes[idx].set_ylabel('Accuracy')
        axes[idx].grid(True, linestyle='--', linewidth=0.5, color='#cccccc')
        axes[idx].legend()

    # Hide any unused axes
    for ax in axes[num_corruptions:]:
        ax.set_visible(False)

    # Make directory to save the plot
    os.makedirs(f"{base_path}/plots", exist_ok=True)

    # Adjusting layout and saving the plot
    plt.tight_layout()
    plt.savefig(f"{base_path}/plots/corruption_accuracies_comparison.png")
    plt.clf()


# # Plot the last epoch accuracies
# plot_last_epoch_accuracies()

# # Plot epoch-wise accuracies for each teacher temperature
# for teT in teacher_temps:
#     plot_epoch_accuracies(teT)

# # Plot the last epoch accuracies for CIFAR-10-C corruptions
plot_last_epoch_accuracies_for_corruptions()
