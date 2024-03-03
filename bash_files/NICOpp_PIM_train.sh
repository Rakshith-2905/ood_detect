#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=23:59:00
#SBATCH --partition=pbatch
#SBATCH --account=ams
#SBATCH --output=./pascal_logs/R-%x.%j.out

domain_name=$1
agg=$2

conda activate unc
cd /usr/workspace/viv41siv/ICML2024/ood_detect/
srun python -u train_mapping_network.py \
            --data_dir './data' \
            --dataset_name NICOpp \
            --domain_name $domain_name \
            --num_classes 60 \
            --batch_size 512 \
            --img_size 224 \
            --seed 42 \
            --task_layer_name model.layer1 \
            --cutmix_alpha 1.0 \
            --warmup_epochs 0 \
            --task_failure_discrepancy_weight 1.0 \
            --task_success_discrepancy_weight 1.5 \
            --attributes_path clip-dissect/NICOpp_core_concepts.json \
            --attributes_embeddings_path data/nicopp/nicopp_core_attributes_CLIP_ViT-B_32_text_embeddings.pth \
            --classifier_name resnet18 \
            --classifier_checkpoint_path "logs/NICOpp/failure_estimation/"$1"/resnet18/classifier/best_checkpoint.pth" \
            --use_imagenet_pretrained \
            --attribute_aggregation $agg \
            --clip_model_name ViT-B/32 \
            --prompt_path data/nicopp/nicopp_CLIP_ViT-B_32_text_embeddings.pth \
            --num_epochs 100 \
            --optimizer adamw \
            --learning_rate 1e-4 \
            --aggregator_learning_rate 1e-3 \
            --scheduler MultiStepLR \
            --val_freq 1 \
            --save_dir ./logs \
            --prefix '' \
            --vlm_dim 512 \
            --num_gpus 1 \
            --num_nodes 1 \
            --augmix_prob 0.2 \
            --cutmix_prob 0.2 