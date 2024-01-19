#!/bin/bash
#SBATCH -N 1
#SBATCH --time=23:55:00
#SBATCH --exclusive
#SBATCH --partition=pbatch
#SBATCH --account=kdml
#SBATCH --qos=standby # added by JEH

# export REQUESTS_CA_BUNDLE=/etc/pki/tls/cert.pem
source activate pytorch

data_dir='./data/'

dataset_name=$1
domain_name=$2
num_classes=$3
img_size=$4
extra_params=$5
classifier_name=$6
classifier_checkpoint_path=$7
prompt_path=$8
num_epochs=$9
a=$9
b=$10
c=$11

# # Print the parameters passed to the script
# echo "data_dir: $data_dir \
#         dataset_name: $dataset_name \
#         domain_name: $domain_name \
#         num_classes: $num_classes \
#         img_size: $img_size \
#         extra_params: $extra_params \
#         classifier_name: $classifier_name \
#         classifier_checkpoint_path: $classifier_checkpoint_path \
#         prompt_path: $prompt_path \
#         num_epochs: $num_epochs \
#         a: $a \
#         b: $b \
#         c: $c"

python train_task_distillation.py \
        --data_dir "$data_dir"  \
        --domain_name "$domain_name"    \
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
        --teacher_temp 2.0  \
        --student_temp 1 \
        --weight_img_loss 1.0  \
        --weight_txt_loss 1.0 \
        --num_gpus 1 \
        --num_nodes 1
