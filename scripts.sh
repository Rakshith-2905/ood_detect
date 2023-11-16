python train_classifier.py --dataset domainnet --domain real --image_size 224 --batch_size 64 \
                            --class_percentage 1     --seed 42     --num_epochs 100     --learning_rate 0.001   
                            --resnet_model resnet50 --use_pretrained

python test_classifier.py --dataset domainnet --domain sketch --image_size 224 --batch_size 64 \
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
            --data_dir 'logs/classifier/resnet50_domainnet_real/features' \
            --domain_name 'real' \
            --dataset_name 'domainnet' \
            --train_on_testset  \
            --use_saved_features \
            --num_classes 345 \
            --batch_size 256 \
            --seed 42 \
            --classifier_name 'resnet50' \
            --classifier_checkpoint_path 'logs/classifier/resnet50_domainnet_real/best_checkpoint.pth' \
            --clip_model_name 'ViT-B/32' \
            --prompt_path 'data/domainnet_v1.0/CLIP_ViT-B-32_text_encodings.pt' \
            --num_epochs 100 --optimizer 'sgd' --learning_rate 0.1 \
            --val_freq 1 \
            --save_dir 'logs/classifier/domainnet/plumber/' \
            --prefix 'scale_100_teT_1_domain_real_lr_0.1_bs256_imgweight_0.5_txtweight_0.5' \
            --proj_clip \
            --projection_dim 512 \
            --teacher_temp 1 \
            --student_temp 1 \
            --weight_img_loss 0.5 \
            --weight_txt_loss 0.5 \
            --num_gpus 1 \
            --num_nodes 1 \
            --is_mlp 



python train_projection_distill_cont.py \
            --data_dir './data/' \
            --domain_name 'real' \
            --dataset_name 'cifar10' \
            --train_on_testset  \
            --num_classes 3 \
            --batch_size 256 \
            --seed 42 \
            --classifier_name 'SimpleCNN' \
            --classifier_checkpoint_path 'cifar10_logs/model_epoch_20.pth' \
            --clip_model_name 'ViT-B/32' \
            --prompt_path 'data/cifar10/CLIP_ViT-B-32_text_encodings.pt' \
            --num_epochs 30 --optimizer 'sgd' --learning_rate 0.1 \
            --val_freq 1 \
            --save_dir 'logs/classifier/cifar10/plumber/' \
            --prefix 'scale_100_epoch20_real_lr_0.1' \
            --proj_clip \
            --projection_dim 512 \
            --teacher_temp 0.5 \
            --student_temp 1 \
            --weight_img_loss 0.0 \
            --weight_txt_loss 1.0 \
            --num_gpus 1 \
            --num_nodes 1



python entropy_viz.py \
        --data_dir './data/' \
        --domain_name 'ID' \
        --dataset_name 'cifar10' \
        --train_on_testset  \
        --num_classes 3 \
        --batch_size 256 \
        --seed 42 \
        --classifier_name 'SimpleCNN' \
        --classifier_checkpoint_path 'cifar10_logs/model_epoch_20.pth' \
        --clip_model_name 'ViT-B/32' \
        --prompt_path 'data/cifar10/CLIP_ViT-B-32_text_encodings.pt' \
        --num_epochs 30 --optimizer 'sgd' --learning_rate 0.1 \
        --val_freq 1 \
        --save_dir 'logs/classifier/cifar10/plumber/' \
        --prefix 'scale_100_epoch20_real_lr_0.1' \
        --proj_clip \
        --projection_dim 512 \
        --teacher_temp 0.5 \
        --student_temp 1 \
        --weight_img_loss 0.0 \
        --weight_txt_loss 1.0 





python cam_masking.py --dataset domainnet --domain real --image_size 224 --batch_size 64 --seed 42  \
                            --resnet_model resnet50 --checkpoint_path 'logs/classifier/resnet50_domainnet_real/best_checkpoint.pth'  \
                            --projector_checkpoint_path 'logs/classifier/resnet50_domainnet_real/projection_default_prompt_gt_sim0_distill1_DN_mapping1/projector_weights.pth' \
                            --resnet_dim 2048 --projection_dim 1024  \
                            --prompt_embeddings_pth "prompts/CLIP_RN50_text_embeddings.pth" --similarity_mode "DN"
