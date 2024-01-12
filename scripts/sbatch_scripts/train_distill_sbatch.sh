#!/bin/bash
#SBATCH -N 1
#SBATCH --time=23:55:00
#SBATCH --exclusive
#SBATCH --partition=pbatch
#SBATCH --account=kdml
#SBATCH --qos=standby # added by JEH

# export REQUESTS_CA_BUNDLE=/etc/pki/tls/cert.pem
source activate pytorch

python legacy_train_classifier.py --batch_size=1 --learning_rate 0.001 --which_epoch=500 --train_epochs 100 --save_freq 1 --use_cuda True \
        --n_context_vectors=8 --token_position middle --learnable_UVTransE True --update_UVTransE True --is_non_linear True --num_predicates=51 \
        --checkpoints_dir_prompt=output/2023-05-09_19-18-07/checkpoints --out_dir=output/2023-05-09_19-18-07 \
        --data_dir='/p/lustre1/rakshith/datasets/SGG/VG/np_files_2' \
        --model VCTree