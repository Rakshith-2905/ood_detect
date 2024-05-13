#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --time=23:59:00
#SBATCH --partition=pbatch
#SBATCH --account=fmi
#SBATCH --gres=gpu:2 
#SBATCH --export=ALL
#SBATCH --output=./pascal_logs/R-%x.%j.out


clear

export MASTER_ADDR=`scontrol show hostname ${SLURM_NODELIST} | head -n1`
export MASTER_PORT=23456
export WORLD_SIZE=16 #1
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE
export CUDA_VISIBLE_DEVICES=0,1

echo "NODELIST="${SLURM_NODELIST}
echo "MASTER_ADDR="$MASTER_ADDR


source activate pytfabric




srun python train_mapping_network.py \
            --data_dir './data' \
            --dataset_name pacs \
            --num_classes 2 \
            --batch_size 64 \
            --img_size 224 \
            --seed 42 \
            --task_layer_name model.encoder.layers.encoder_layer_1 \
            --cutmix_alpha 1.0 \
            --warmup_epochs 0 \
            --task_failure_discrepancy_weight 2.0 \
            --task_success_discrepancy_weight 1.5 \
            --attributes_path clip-dissect/pacs_core_concepts_25_corrupted.json \
            --attributes_embeddings_path data/pacs/pacs_25_corr_attributes_CLIP_ViT-B_32_text_embeddings.pth \
            --classifier_name resnet18 \
            --classifier_checkpoint_path logs/pacs-photo/resnet18/classifier/best_checkpoint.pth \
            --use_imagenet_pretrained \
            --attribute_aggregation mean \
            --clip_model_name ViT-B/32 \
            --prompt_path data/pacs/pacs_CLIP_ViT-B_32_text_embeddings.pth \
            --num_epochs 200 \
            --optimizer adamw \
            --learning_rate 1e-3 \
            --aggregator_learning_rate 1e-3 \
            --scheduler MultiStepLR \
            --val_freq 1 \
            --save_dir ./logs \
            --prefix '25_corr' \
            --vlm_dim 512 \
            --num_gpus 2 \
            --num_nodes 1 \
            --augmix_prob 0.2 \
            --cutmix_prob 0.2 

# srun python train_mapping_network.py \
#             --data_dir './data' \
#             --dataset_name domainnet \
#             --domain_name 'painting' \
#             --num_classes 345 \
#             --batch_size 64 \
#             --img_size 224 \
#             --seed 42 \
#             --task_layer_name model.layer1 \
#             --cutmix_alpha 1.0 \
#             --warmup_epochs 0 \
#             --task_failure_discrepancy_weight 2.0 \
#             --task_success_discrepancy_weight 1.5 \
#             --attributes_path clip-dissect/domainnet_core_concepts.json \
#             --attributes_embeddings_path data/domainnet_v1.0/domainnet_attributes_CLIP_ViT-B_32_text_embeddings.pth \
#             --classifier_name resnet18 \
#             --classifier_checkpoint_path logs/domainnet-real/resnet18/classifier/checkpoint_29.pth \
#             --use_imagenet_pretrained \
#             --attribute_aggregation mean \
#             --clip_model_name ViT-B/32 \
#             --prompt_path data/domainnet_v1.0/domainnet_CLIP_ViT-B_32_text_embeddings.pth \
#             --num_epochs 200 \
#             --optimizer adamw \
#             --learning_rate 1e-3 \
#             --aggregator_learning_rate 1e-3 \
#             --scheduler MultiStepLR \
#             --val_freq 1 \
#             --save_dir ./logs \
#             --prefix '' \
#             --vlm_dim 512 \
#             --num_gpus 2 \
#             --num_nodes 1 \
#             --augmix_prob 0.2 \
#             --cutmix_prob 0.2 

# python train_mapping_network.py \
#             --data_dir './data' \
#             --dataset_name Waterbirds \
#             --num_classes 2 \
#             --batch_size 64 \
#             --img_size 224 \
#             --seed 42 \
#             --task_layer_name model.encoder.layers.encoder_layer_1 \
#             --cutmix_alpha 1.0 \
#             --warmup_epochs 0 \
#             --task_failure_discrepancy_weight 2.0 \
#             --task_success_discrepancy_weight 1.5 \
#             --attributes_path clip-dissect/Waterbirds_core_concepts.json \
#             --attributes_embeddings_path data/Waterbirds/Waterbirds_attributes_CLIP_ViT-B_32_text_embeddings.pth \
#             --classifier_name vit_b_16 \
#             --classifier_checkpoint_path logs/Waterbirds/failure_estimation/None/vit_b_16/classifier_seed42/checkpoint_99.pth \
#             --use_imagenet_pretrained \
#             --attribute_aggregation mean \
#             --clip_model_name ViT-B/32 \
#             --prompt_path data/Waterbirds/Waterbirds_CLIP_ViT-B_32_text_embeddings.pth \
#             --num_epochs 200 \
#             --optimizer adamw \
#             --learning_rate 1e-3 \
#             --aggregator_learning_rate 1e-3 \
#             --scheduler MultiStepLR \
#             --val_freq 1 \
#             --save_dir ./logs \
#             --prefix '' \
#             --vlm_dim 512 \
#             --num_gpus 2 \
#             --num_nodes 1 \
#             --augmix_prob 0.2 \
#             --cutmix_prob 0.2 

# python train_mapping_network.py \
#             --data_dir './data' \
#             --dataset_name CelebA \
#             --num_classes 2 \
#             --batch_size 64 \
#             --img_size 224 \
#             --seed 42 \
#             --task_layer_name model.encoder.layers.encoder_layer_9 \
#             --cutmix_alpha 1.0 \
#             --warmup_epochs 0 \
#             --task_failure_discrepancy_weight 2.0 \
#             --task_success_discrepancy_weight 1.5 \
#             --attributes_path clip-dissect/CelebA_core_concepts.json \
#             --attributes_embeddings_path data/CelebA/CelebA_attributes_CLIP_ViT-B_32_text_embeddings.pth \
#             --classifier_name vit_b_16 \
#             --classifier_checkpoint_path logs/CelebA/failure_estimation/None/vit_b_16/classifier_seed42/checkpoint_20.pth \
#             --use_imagenet_pretrained \
#             --attribute_aggregation max \
#             --clip_model_name ViT-B/32 \
#             --prompt_path data/CelebA/CelebA_CLIP_ViT-B_32_text_embeddings.pth \
#             --num_epochs 200 \
#             --optimizer adamw \
#             --learning_rate 1e-3 \
#             --aggregator_learning_rate 1e-3 \
#             --scheduler MultiStepLR \
#             --val_freq 1 \
#             --save_dir ./logs \
#             --prefix '' \
#             --vlm_dim 512 \
#             --num_gpus 2 \
#             --num_nodes 1 \
#             --augmix_prob 0.2 \
#             --cutmix_prob 0.2 


# python train_mapping_network.py \
#             --data_dir './data' \
#             --dataset_name cats_dogs \
#             --num_classes 2 \
#             --batch_size 64 \
#             --img_size 224 \
#             --seed 42 \
#             --task_layer_name model.encoder.layers.encoder_layer_1 \
#             --cutmix_alpha 1.0 \
#             --warmup_epochs 0 \
#             --task_failure_discrepancy_weight 2.0 \
#             --task_success_discrepancy_weight 1.5 \
#             --attributes_path clip-dissect/cats_dogs_core_concepts.json \
#             --attributes_embeddings_path data/cats_dogs/cats_dogs_attributes_CLIP_ViT-B_32_text_embeddings.pth \
#             --classifier_name vit_b_16 \
#             --classifier_checkpoint_path logs/cats_dogs/vit_b_16/classifier_seed42/checkpoint_99.pth \
#             --use_imagenet_pretrained \
#             --attribute_aggregation mean \
#             --clip_model_name ViT-B/32 \
#             --prompt_path data/cats_dogs/cats_dogs_CLIP_ViT-B_32_text_embeddings.pth \
#             --num_epochs 200 \
#             --optimizer adamw \
#             --learning_rate 1e-3 \
#             --aggregator_learning_rate 1e-3 \
#             --scheduler MultiStepLR \
#             --val_freq 1 \
#             --save_dir ./logs \
#             --prefix '' \
#             --vlm_dim 512 \
#             --num_gpus 2 \
#             --num_nodes 1 \
#             --augmix_prob 0.2 \
#             --cutmix_prob 0.2 



# python failure_detection_eval.py \
#     --data_dir './data' \
#     --dataset_name Waterbirds \
#     --num_classes 2 \
#     --batch_size 64 \
#     --img_size 32 \
#     --seed 42 \
#     --task_layer_name model.encoder.layers.encoder_layer_1 \
#     --cutmix_alpha 1.0 \
#     --warmup_epochs 0 \
#     --attributes_path clip-dissect/Waterbirds_core_concepts.json \
#     --attributes_embeddings_path data/Waterbirds/Waterbirds_attributes_CLIP_ViT-B_32_text_embeddings.pth \
#     --classifier_name vit_b_16 \
#     --classifier_checkpoint_path logs/Waterbirds/failure_estimation/None/vit_b_16/classifier_seed42/checkpoint_99.pth \
#     --use_imagenet_pretrained \
#     --attribute_aggregation max \
#     --clip_model_name ViT-B/32 \
#     --prompt_path data/Waterbirds/Waterbirds_CLIP_ViT-B_32_text_embeddings.pth \
#     --num_epochs 200 \
#     --optimizer adamw \
#     --learning_rate 1e-3 \
#     --aggregator_learning_rate 1e-3 \
#     --scheduler MultiStepLR \
#     --val_freq 1 \
#     --save_dir ./logs \
#     --prefix '' \
#     --vlm_dim 512 \
#     --num_gpus 1 \
#     --num_nodes 1 \
#     --augmix_prob 0.2 \
#     --cutmix_prob 0.2 \
#     --resume_checkpoint_path logs/Waterbirds/failure_estimation/None/vit_b_16/mapper/_agg_max_bs_64_lr_0.001_augmix_prob_0.2_cutmix_prob_0.2_scheduler_warmup_epoch_0_layer_model.encoder.layers.encoder_layer_1/pim_weights_best.pth \
#     --method baseline \
#     --score pe 

