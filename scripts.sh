python train_classifier.py --dataset domainnet --domain real --image_size 224 --batch_size 64 \
                            --class_percentage 1     --seed 42     --num_epochs 100     --learning_rate 0.001   
                            --resnet_model resnet50 --use_pretrained

python test_classifier.py --dataset domainnet --domain real --image_size 224 --batch_size 64 \
                            --class_percentage 1     --seed 42  \
                            --resnet_model resnet50 --checkpoint_path 'logs/classifier/resnet50_domainnet_real/best_checkpoint.pth'

python cam_masking.py --model_path 'logs/classifier/resnet_resnet18/best_model_weights.pth' \
                      --resnet_model resnet18 \
                      --dataset domainnet

python train_projection.py --dataset domainnet --domain real --image_size 224 --batch_size 32 --seed 42  \
                            --resnet_model resnet50 --checkpoint_path 'logs/classifier/resnet50_domainnet_real/best_checkpoint.pth' \
                            --resnet_dim 2048 --projection_dim 512 --teacher_temp 0.5 --student_temp 0.1 \
                            --prompt_embeddings_pth "prompts/text_embeddings.pth" --use_default_prompt True \
                            --similarity_mode "cosine"
