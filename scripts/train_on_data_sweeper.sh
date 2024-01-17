#!/bin/bash
# Function to generate extra parameters based on type and txt_prompt_type
generate_extra_params() {
    local type=$1
    local txt_prompt_type=$2
    local extra_params=""
    local model_type=""

    case "$type" in
        "A") 
            extra_params="--img_prompting"
            extra_params+=" $([ "$txt_prompt_type" = "class" ] && echo "--cls_txt_prompts" || echo "--dataset_txt_prompt")"
            model_type="$([ "$txt_prompt_type" = "class" ] && echo "plumber_img_prompt_cls_LP" || echo "plumber_img_prompt_dataset_LP")"
            ;;
        "B") 
            extra_params="--img_prompting --txt_projection"
            model_type="plumber_text_proj_img_prompt"
            ;;
        "C") 
            extra_params="--img_prompting --txt_projection"
            extra_params+=" $([ "$txt_prompt_type" = "class" ] && echo "--cls_txt_prompts" || echo "--dataset_txt_prompt")"
            model_type="$([ "$txt_prompt_type" = "class" ] && echo "plumber_text_proj_img_prompt_cls_LP" || echo "plumber_text_proj_img_prompt_dataset_LP")"
            ;;
        "D") 
            extra_params="--img_projection"
            extra_params+=" $([ "$txt_prompt_type" = "class" ] && echo "--cls_txt_prompts" || echo "--dataset_txt_prompt")"
            model_type="$([ "$txt_prompt_type" = "class" ] && echo "plumber_img_proj_cls_LP" || echo "plumber_img_proj_dataset_LP")"
            ;;
        "E") 
            extra_params="--img_projection --txt_projection"
            model_type="plumber_img_text_proj"
            ;;
        "F") 
            extra_params="--img_projection --txt_projection"
            extra_params+=" $([ "$txt_prompt_type" = "class" ] && echo "--cls_txt_prompts" || echo "--dataset_txt_prompt")"
            model_type="$([ "$txt_prompt_type" = "class" ] && echo "plumber_img_text_proj_cls_LP" || echo "plumber_img_text_proj_dataset_LP")"
            ;;
        "G") 
            extra_params="--img_projection --img_prompting"
            extra_params+=" $([ "$txt_prompt_type" = "class" ] && echo "--cls_txt_prompts" || echo "--dataset_txt_prompt")"
            model_type="$([ "$txt_prompt_type" = "class" ] && echo "plumber_img_proj_img_prompt_cls_LP" || echo "plumber_img_proj_img_prompt_dataset_LP")"
            ;;
        "H") 
            extra_params="--img_projection --img_prompting --txt_projection"
            model_type="plumber_img_text_proj_img_prompt"
            ;;
        "I") 
            extra_params="--img_projection --img_prompting"
            extra_params+=" $([ "$txt_prompt_type" = "class" ] && echo "--cls_txt_prompts --txt_projection" || echo "--dataset_txt_prompt --txt_projection")"
            model_type="$([ "$txt_prompt_type" = "class" ] && echo "plumber_img_text_proj_img_prompt_cls_LP" || echo "plumber_img_text_proj_img_prompt_dataset_LP")"
            ;;
        "J")
            extra_params="--img_prompting"
            model_type="plumber_img_prompt"
            ;;
        "K")
            extra_params="--img_projection"
            model_type="plumber_img_proj"
            ;;
        "L")
            extra_params="--img_projection --img_prompting"
            model_type="plumber_img_proj_img_prompt"
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
    echo "$model_type"
    
    return 0
}

# Define arrays for parameters
# teacher_temps=(0.5 2 5 10)
# weight_img_loss=(0.5 1)
# weight_txt_loss=(0.5 1 2 4 8)

teacher_temps=(2.0)
weight_img_loss=(1.0)
weight_txt_loss=(1.0)


dataset_name='NICOpp' # (CelebA, Waterbirds, cifar10, cifar10-limited)
classifier_name='resnet18' # (resnet18, resnet50, resnet101, SimpleCNN)
num_epochs=10
data_dir='./data/'
classifier_checkpoint_path="logs/${dataset_name}/${classifier_name}/classifier/checkpoint_29.pth"
prompt_path="data/${dataset_name}/${dataset_name}_CLIP_ViT-B_32_text_embeddings.pth"
img_size=224 # only for celeba
num_classes=60
# Class level or dataset level text prompts
txt_prompt_type="dataset" # Options: class, dataset

# List for attributes
atts=(4)

# types=("A" "B" "C" "D" "E" "F" "G" "H" "I", "J", "K", "L")

# types=("A" "B" "C" "J" "K" "L")
types=("A" "B" "C" "E" "J" "K" "L" "M" "N" "O")


for type in "${types[@]}"; do
    output=$(generate_extra_params "$type" "$txt_prompt_type")
    extra_params=$(echo "$output" | head -n 1)
    model_type=$(echo "$output" | tail -n 1)

    echo "Running type: $type with params: $extra_params and model type: $model_type"
    
    for a in "${teacher_temps[@]}"; do
        for b in "${weight_img_loss[@]}"; do
            for c in "${weight_txt_loss[@]}"; do
                python train_on_data.py \
                        --data_dir "$data_dir"  \
                        --domain_name 'real'    \
                        --dataset_name "$dataset_name"    \
                        --attributes "$atts"  \
                        --num_classes "$num_classes"  \
                        --batch_size 256  \
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
