#!/bin/bash

# Function to generate extra parameters based on type and txt_prompt_type
generate_extra_params() {
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


dataset_name='Waterbirds' # (CelebA, Waterbirds, cifar10, cifar10-limited)
classifier_name='resnet18' # (resnet18, resnet50, resnet101, SimpleCNN)
num_epochs=10
data_dir='./data/'
classifier_checkpoint_path="logs/${dataset_name}/${classifier_name}/classifier/checkpoint_99.pth"
prompt_path="data/${dataset_name}/${dataset_name}_CLIP_ViT-B_32_text_embeddings.pth"
img_size=224 # only for celeba
num_classes=2
# Class level or dataset level text prompts
txt_prompt_type="class" # Options: class, dataset

# types=("E")
types=("J" "K" "L")



for type in "${types[@]}"; do
    extra_params=$(generate_extra_params "$type" "$txt_prompt_type")
    [[ $? -ne 0 ]] && exit 1

    echo "Running type: $type with params: $extra_params"
    for a in "${teacher_temps[@]}"; do
        for b in "${weight_img_loss[@]}"; do
            for c in "${weight_txt_loss[@]}"; do
                python train_task_distillation.py \
                        --data_dir "$data_dir"  \
                        --domain_name 'real'    \
                        --dataset_name "$dataset_name"    \
                        --train_on_testset    \
                        --num_classes "$num_classes"  \
                        --batch_size 128  \
                        --seed 42    \
                        --img_size "$img_size"  \
                        $extra_params \
                        --classifier_name "$classifier_name" \
                        --classifier_checkpoint_path "$classifier_checkpoint_path" \
                        --clip_model_name 'ViT-B/32' \
                        --prompt_path "$prompt_path" \
                        --n_promt_ctx 16 \
                        --num_epochs "$num_epochs" \
                        --optimizer 'sgd' \
                        --learning_rate 0.1 \
                        --val_freq 1 \
                        --prefix '' \
                        --proj_clip \
                        --projection_dim 512 \
                        --teacher_temp $a  \
                        --student_temp 1 \
                        --weight_img_loss $b  \
                        --weight_txt_loss $c \
                        --num_gpus 1 \
                        --num_nodes 1
            done
        done
    done
done
