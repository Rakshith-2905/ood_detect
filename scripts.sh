# Training ResNet model on domainnet dataset
python train_classifier.py --dataset domainnet --split train --image_size 224 --batch_size 32 \
                           --class_percentage 0.5     --seed 42     --num_epochs 10     --learning_rate 0.001 \
                           --resnet_model resnet18     --use_pretrained