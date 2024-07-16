
# python failure_detection_eval.py \
#     --data_dir './data' \
#     --dataset_name Waterbirds \
#     --num_classes 2 \
#     --batch_size 64 \
#     --img_size 32 \
#     --seed 42 \
#     --task_layer_name model.encoder.layers.encoder_layer_9 \
#     --cutmix_alpha 1.0 \
#     --warmup_epochs 0 \
#     --attributes_path clip-dissect/Waterbirds_core_concepts.json \
#     --attributes_embeddings_path data/Waterbirds/Waterbirds_attributes_CLIP_ViT-B_32_text_embeddings.pth \
#     --classifier_name vit_b_16 \
#     --classifier_checkpoint_path logs/Waterbirds/failure_estimation/None/vit_b_16/classifier_seed42/checkpoint_99.pth \
#     --use_imagenet_pretrained \
#     --attribute_aggregation mean \
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
#     --resume_checkpoint_path logs/Waterbirds/failure_estimation/None/vit_b_16/mapper/_agg_mean_bs_128_lr_0.001_augmix_prob_0.2_cutmix_prob_0.2_scheduler_warmup_epoch_0_layer_model.encoder.layers.encoder_layer_9/pim_weights_final.pth \
#     --method pim \
#     --score cross_entropy



# python failure_detection_eval.py \
#     --data_dir './data' \
#     --dataset_name Waterbirds \
#     --num_classes 2 \
#     --batch_size 64 \
#     --img_size 32 \
#     --seed 42 \
#     --task_layer_name model.encoder.layers.encoder_layer_9 \
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
#     --resume_checkpoint_path logs/Waterbirds/failure_estimation/None/vit_b_16/mapper/_agg_max_bs_64_lr_0.001_augmix_prob_0.2_cutmix_prob_0.2_scheduler_warmup_epoch_0_layer_model.encoder.layers.encoder_layer_9/pim_weights_best.pth \
#     --method pim \
#     --score cross_entropy



python failure_detection_eval.py \
    --data_dir './data' \
    --dataset_name cats_dogs \
    --num_classes 2 \
    --batch_size 64 \
    --img_size 32 \
    --seed 42 \
    --task_layer_name model.encoder.layers.encoder_layer_1 \
    --cutmix_alpha 1.0 \
    --warmup_epochs 0 \
    --attributes_path clip-dissect/cats_dogs_core_concepts.json \
    --attributes_embeddings_path data/cats_dogs/cats_dogs_attributes_CLIP_ViT-B_32_text_embeddings.pth \
    --classifier_name vit_b_16 \
    --classifier_checkpoint_path logs/cats_dogs/vit_b_16/classifier_seed42/checkpoint_99.pth \
    --use_imagenet_pretrained \
    --attribute_aggregation mean \
    --clip_model_name ViT-B/32 \
    --prompt_path data/cats_dogs/cats_dogss_CLIP_ViT-B_32_text_embeddings.pth \
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
    --resume_checkpoint_path logs/cats_dogs/vit_b_16/mapper/_agg_mean_bs_64_lr_0.001_augmix_prob_0.2_cutmix_prob_0.2_scheduler_warmup_epoch_0_layer_model.encoder.layers.encoder_layer_1/pim_weights_best.pth \
    --method pim \
    --score cross_entropy





python failure_detection_eval.py \
    --data_dir './data' \
    --dataset_name imagenet_a \
    --num_classes 1000 \
    --batch_size 64 \
    --img_size 32 \
    --seed 42 \
    --task_layer_name model.layer4 \
    --cutmix_alpha 1.0 \
    --warmup_epochs 0 \
    --attributes_path clip-dissect/imagenet_core_concepts.json \
    --attributes_embeddings_path data/imagenet/imagenet_core_attributes_CLIP_ViT-B_32_text_embeddings.pth \
    --classifier_name resnet50 \
    --classifier_checkpoint_path logs/imagenet/resnet50/classifier/non_existing_path.pth \
    --use_imagenet_pretrained \
    --attribute_aggregation mean \
    --clip_model_name ViT-B/32 \
    --prompt_path data/imagenet/imagenet_CLIP_ViT-B_32_text_embeddings.pth \
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
    --resume_checkpoint_path logs/imagenet/resnet50/mapper/_agg_mean_bs_64_lr_0.001_augmix_prob_0.2_cutmix_prob_0.2_scheduler_warmup_epoch_0_layer_model.layer4/pim_weights_166.pth \
    --method pim \
    --score cross_entropy