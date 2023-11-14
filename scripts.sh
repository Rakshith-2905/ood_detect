python train_classifier.py --dataset domainnet --domain real --image_size 224 --batch_size 64 \
                            --class_percentage 1     --seed 42     --num_epochs 100     --learning_rate 0.001   
                            --resnet_model resnet50 --use_pretrained

python test_classifier.py --dataset domainnet --domain clipart --image_size 224 --batch_size 64 \
                            --class_percentage 1     --seed 42  \
                            --resnet_model resnet50 --checkpoint_path 'logs/classifier/resnet50_domainnet_real/best_checkpoint.pth'

python cam_masking.py --model_path 'logs/classifier/resnet_resnet18/best_model_weights.pth' \
                      --resnet_model resnet18 \
                      --dataset domainnet

python train_projection.py --dataset domainnet --domain real --image_size 224 --batch_size 64 --seed 42  \
                            --resnet_model resnet50 --checkpoint_path 'logs/classifier/resnet50_domainnet_real/best_checkpoint.pth' \
                            --resnet_dim 2048 --projection_dim 1024 --teacher_temp 0.5 --student_temp 1 \
                            --prompt_embeddings_pth "prompts/CLIP_RN50_text_embeddings.pth" --use_default_prompt True \
                            --similarity_mode "DN" --feature_sim_weight 0.1

############################### 1. Test Scripts Projection with default prompt #########################################

# DN Similarity

python test_projection.py --dataset domainnet --domain real --image_size 224 --batch_size 64 --seed 42  \
                            --resnet_model resnet50 --checkpoint_path 'logs/classifier/resnet50_domainnet_real/best_checkpoint.pth'  \
                            --projector_checkpoint_path 'logs/classifier/resnet50_domainnet_real/projection_default_prompt_gt_sim0_distill1_DN_mapping1/projector_weights.pth' \
                            --resnet_dim 2048 --projection_dim 1024  \
                            --prompt_embeddings_pth "prompts/CLIP_RN50_text_embeddings.pth" --similarity_mode "DN"
# Cosine Similarity

python test_projection.py --dataset domainnet --domain real --image_size 224 --batch_size 64 --seed 42  \
                            --resnet_model resnet50 --checkpoint_path 'logs/classifier/resnet50_domainnet_real/best_checkpoint.pth'  \
                            --projector_checkpoint_path 'logs/classifier/resnet50_domainnet_real/projection_default_prompt_gt_sim0_distill1_cosine_mapping1/projector_weights.pth' \
                            --resnet_dim 2048 --projection_dim 1024  \
                            --prompt_embeddings_pth "prompts/CLIP_RN50_text_embeddings.pth" --similarity_mode "cosine"

################################ Contrastive PLUMBER #########################################

python train_projection_distill_cont.py \
            --data_dir 'logs/classifier/resnet50_domainnet_real/features/clipart' \
            --domain_name 'clipart' \
            --dataset_name 'domainnet' \
            --train_on_testset  \
            --use_saved_features \
            --num_classes 345 \
            --batch_size 32 \
            --seed 42 \
            --classifier_name 'resnet50' \
            --classifier_checkpoint_path 'logs/classifier/resnet50_domainnet_real/best_checkpoint.pth' \
            --clip_model_name 'ViT-B/32' \
            --prompt_path 'data/domainnet_v1.0/CLIP_ViT-B-32_text_encodings.pt' \
            --num_epochs 100 --optimizer 'sgd' --learning_rate 0.1 \
            --val_freq 1 \
            --save_dir 'logs/classifier/domainnet/plumber/' \
            --prefix 'domain_clipart_lr_0.1' \
            --proj_clip \
            --projection_dim 512 \
            --teacher_temp 0.5 \
            --student_temp 1 \
            --num_gpus 1 \
            --num_nodes 1 \



# GRAD CAM MASKING

python cam_masking.py --dataset domainnet --domain real --image_size 224 --batch_size 64 --seed 42  \
                            --resnet_model resnet50 --checkpoint_path 'logs/classifier/resnet50_domainnet_real/best_checkpoint.pth'  \
                            --projector_checkpoint_path 'logs/classifier/resnet50_domainnet_real/projection_default_prompt_gt_sim0_distill1_DN_mapping1/projector_weights.pth' \
                            --resnet_dim 2048 --projection_dim 1024  \
                            --prompt_embeddings_pth "prompts/CLIP_RN50_text_embeddings.pth" --similarity_mode "DN"
