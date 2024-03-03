#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:59:00
#SBATCH --partition=pbatch
#SBATCH --account=ams
#SBATCH --output=./pascal_logs/R-%x.%j.out

train_domain_name=$1
agg=$2
method=$3

conda activate unc
cd /usr/workspace/viv41siv/ICML2024/ood_detect/
srun python -u failure_detection_eval.py \
        --data_dir './data' \
        --dataset_name NICOpp \
        --eval_dataset NICOpp \
        --num_classes 60 \
        --batch_size 512 \
        --img_size 224 \
        --seed 42 \
        --task_layer_name model.layer1 \
        --cutmix_alpha 1.0 \
        --warmup_epochs 10 \
        --attributes_path clip-dissect/NICOpp_core_concepts.json \
        --attributes_embeddings_path data/nicopp/nicopp_core_attributes_CLIP_ViT-B_32_text_embeddings.pth \
        --classifier_name resnet18 \
        --classifier_checkpoint_path "logs/NICOpp/failure_estimation/"$train_domain_name"/resnet18/classifier/best_checkpoint.pth" \
        --use_imagenet_pretrained \
        --attribute_aggregation $agg \
        --clip_model_name ViT-B/32 \
        --prompt_path data/nicopp/nicopp_core_attributes_CLIP_ViT-B_32_text_embeddings.pth \
        --num_epochs 200 \
        --optimizer adamw \
        --learning_rate 1e-3 \
        --aggregator_learning_rate 1e-3 \
        --scheduler MultiStepLR \
        --val_freq 1 \
        --save_dir ./logs \
        --prefix '' \
        --vlm_dim 512 \
        --num_gpus 1 \
        --num_nodes 1 \
        --augmix_prob 0.2 \
        --cutmix_prob 0.2 \
        --resume_checkpoint_path "logs/NICOpp/resnet18/"$train_domain_name"/mapper/_agg_"$agg"_bs_512_lr_0.0001_augmix_prob_0.2_cutmix_prob_0.2_scheduler_warmup_epoch_0_layer_model.layer1/pim_weights_final.pth" \
        --method $method \
        --score cross_entropy 