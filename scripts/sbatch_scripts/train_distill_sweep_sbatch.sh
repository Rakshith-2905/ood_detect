#!/bin/bash

# Function to generate extra parameters based on type and txt_prompt_type
generate_extra_params() {
    # Mapping of 'type' to 'model_type' when 'txt_prompt_type' is set to "class":
    # A -> plumber_img_prompt_cls_LP
    # B -> plumber_text_proj_img_prompt
    # C -> plumber_text_proj_img_prompt_cls_LP
    # D -> plumber_img_proj_cls_LP
    # E -> plumber_img_text_proj
    # F -> plumber_img_text_proj_cls_LP
    # G -> plumber_img_proj_img_prompt_cls_LP
    # H -> plumber_img_text_proj_img_prompt
    # I -> plumber_img_text_proj_img_prompt_cls_LP
    # J -> plumber_img_prompt
    # K -> plumber_img_proj
    # L -> plumber_img_proj_img_prompt
    # M -> plumber_text_proj
    # N -> plumber_cls_LP
    # O -> plumber_text_proj_cls_LP
    local type=$1
    local txt_prompt_type=$2
    local extra_params=""

    case "$type" in
        "A") 
            extra_params="--img_prompting"
            extra_params+=" $([ "$txt_prompt_type" = "class" ] && echo "--cls_txt_prompts" || echo "--dataset_txt_prompt")"
            ;;
        "B") 
            extra_params="--img_prompting --txt_projection"
            ;;
        "C") 
            extra_params="--img_prompting --txt_projection"
            extra_params+=" $([ "$txt_prompt_type" = "class" ] && echo "--cls_txt_prompts" || echo "--dataset_txt_prompt")"
            ;;
        "D") 
            extra_params="--img_projection"
            extra_params+=" $([ "$txt_prompt_type" = "class" ] && echo "--cls_txt_prompts" || echo "--dataset_txt_prompt")"
            ;;
        "E") 
            extra_params="--img_projection --txt_projection"
            ;;
        "F") 
            extra_params="--img_projection --txt_projection"
            extra_params+=" $([ "$txt_prompt_type" = "class" ] && echo "--cls_txt_prompts" || echo "--dataset_txt_prompt")"
            ;;
        "G") 
            extra_params="--img_projection --img_prompting"
            extra_params+=" $([ "$txt_prompt_type" = "class" ] && echo "--cls_txt_prompts" || echo "--dataset_txt_prompt")"
            ;;
        "H") 
            extra_params="--img_projection --img_prompting --txt_projection"
            ;;
        "I") 
            extra_params="--img_projection --img_prompting"
            extra_params+=" $([ "$txt_prompt_type" = "class" ] && echo "--cls_txt_prompts --txt_projection" || echo "--dataset_txt_prompt --txt_projection")"
            ;;
        "J")
            extra_params="--img_prompting"
            ;;
        "K")
            extra_params="--img_projection"
            ;;
        "L")
            extra_params="--img_projection --img_prompting"
            ;;
        "M")
            extra_params=" --txt_projection"
            model_type="plumber_text_proj"
        ;;
        "N")
            extra_params+=" $([ "$txt_prompt_type" = "class" ] && echo "--cls_txt_prompts" || echo "--dataset_txt_prompt")"
            model_type="$([ "$txt_prompt_type" = "class" ] && echo "plumber_cls_LP" || echo "plumber_dataset_LP")"
        ;;
        "O")
            extra_params=" --txt_projection"
            extra_params+=" $([ "$txt_prompt_type" = "class" ] && echo "--cls_txt_prompts --txt_projection" || echo "--dataset_txt_prompt --txt_projection")"
            model_type="$([ "$txt_prompt_type" = "class" ] && echo "plumber_text_proj_cls_LP" || echo "plumber_text_proj_dataset_LP")"
        ;;
        *)
            echo "Invalid type"
            return 1
            ;;
    esac
    echo "$extra_params"
    return 0
}

# Define arrays for parameters
# teacher_temps=(0.5 2 5 10)
# weight_img_loss=(0.5 1)
# weight_txt_loss=(0.5 1 2 4 8)

teacher_temps=(2)
weight_img_loss=(1)
weight_txt_loss=(1)


dataset_name='domainnet' # (CelebA, Waterbirds, cifar10, cifar10-limited, cifar100, cifar100-90cls, domainnet)
domain_name='real' # (real, sketch, quickdraw, infograph, painting, clipart)
classifier_name='resnet18' # (resnet18, resnet50, resnet101, SimpleCNN)
num_epochs=100
data_dir='./data/'

classifier_checkpoint_path="logs/domainnet-real/resnet18/classifier/checkpoint_29.pth"
# classifier_checkpoint_path="logs/${dataset_name}/${classifier_name}/classifier/checkpoint_99.pth"
prompt_path="data/${dataset_name}/${dataset_name}_CLIP_ViT-B_32_text_embeddings.pth"

img_size=224 # only for celeba
num_classes=345
# Class level or dataset level text prompts
txt_prompt_type="class" # Options: class, dataset

types=("A" "B" "C" "D" "E" "F" "G" "H" "I" "J" "K" "L")


for type in "${types[@]}"; do
    extra_params=$(generate_extra_params "$type" "$txt_prompt_type")
    [[ $? -ne 0 ]] && exit 1

    echo "Running type: $type with params: $extra_params"
    for a in "${teacher_temps[@]}"; do
        for b in "${weight_img_loss[@]}"; do
            for c in "${weight_txt_loss[@]}"; do
                # echo "data_dir: $data_dir \
                #     dataset_name: $dataset_name \
                #     domain_name: $domain_name \
                #     num_classes: $num_classes \
                #     img_size: $img_size \
                #     extra_params: $extra_params \
                #     classifier_name: $classifier_name \
                #     classifier_checkpoint_path: $classifier_checkpoint_path \
                #     prompt_path: $prompt_path \
                #     num_epochs: $num_epochs \
                #     a: $a \
                #     b: $b \
                #     c: $c" 
                sbatch scripts/sbatch_scripts/train_distill_sbatch.sh \
                    "$dataset_name" \
                    "$domain_name" \
                    "$num_classes" \
                    "$img_size" \
                    "$extra_params" \
                    "$classifier_name" \
                    "$classifier_checkpoint_path" \
                    "$prompt_path" \
                    "$num_epochs" \
                    "$a" \
                    "$b" \
                    "$c"
            done
        done
    done
done
