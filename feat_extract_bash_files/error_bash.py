"""
A simple code to generate the bash files for submitting jobs
"""
import os
import sys
import pandas as pd

train_chunks = pd.read_csv("/g/g92/thopalli/ws_kdml/KDML/ood_detect/error_files_train.txt", sep='\t')
test_chunks = pd.read_csv("/g/g92/thopalli/ws_kdml/KDML/ood_detect/error_files_test.txt", sep='\t')
train_json_file= "/usr/workspace/KDML/yfcc/yfcc15m_clean_open_data_already_downloaded_train_final.json"
test_json_file= "/usr/workspace/KDML/yfcc/yfcc15m_clean_open_data_already_downloaded_test_final.json"
model=['dino_vits16']
#reead the train and test chunks

index_files={}
index_files['train']=train_chunks
index_files['test']=test_chunks
json_files={}
json_files['train']=train_json_file
json_files['test']=test_json_file

save_base_dir='feat_extract_bash_files_scripts_error'

os.makedirs(save_base_dir,exist_ok=True)


modes=['train','test']
for m in model:
    save_file_dir= os.path.join(save_base_dir)
    log_dir= os.path.join(f'/usr/workspace/KDML/ood_detect/feat_extract_bash_files/feat_logs_errors')
    os.makedirs(save_file_dir,exist_ok=True)
    os.makedirs(log_dir,exist_ok=True)
    for mode in modes:
        json_file= json_files[mode]
        index_file= index_files[mode]
        for chunk_i in range(len(index_file)):
            start_index= index_file['start_indices'][chunk_i]
            end_index= index_file['end_indices'][chunk_i]
            

            filename= os.path.join(save_file_dir,f'{m}/{mode}/split_{chunk_i}.sh')
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            out_file= os.path.join(log_dir,f'{m}/{mode}/split_{chunk_i}.out')
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            with open(filename, "w") as f:

                f.write("#!/bin/sh\n")
                f.write("#BSUB -alloc_flags ipisolate\n")
                f.write("#BSUB -nnodes 1\n")
                f.write("#BSUB -q expedite\n")
                f.write("#BSUB -G iebdl\n")
                f.write("#BSUB -U wedat\n")
                f.write(f"#BSUB -o {out_file}\n\n")

                f.write("cd /usr/workspace/KDML/\n")
                f.write("source opence1.9/anaconda/bin/activate\n")
                f.write("module load cuda/11.8\n")
                f.write("conda activate opence-1.9.1\n")
                f.write("export HF_HOME='/usr/workspace/KDML/.cache/huggingface'\n")
                f.write("export HF_DATASETS_CACHE='/usr/workspace/KDML/.cache/huggingface/datasets'\n")
                f.write("export JSM_NAMESPACE_SIZE=512\n")
                f.write("export OMP_NUM_THREADS=10\n")
                f.write("cd /usr/workspace/KDML/ood_detect\n")

                f.write(f'lrun -n 4 -N 1 python nothing_feat_extract.py --json_file {json_file} --feature_extractor_name {m} --clip_model_name ViT-B/32 --save_path "/p/gpfs1/KDML/feats/{mode}" --data_path "./data" --batch_size 1024 --images_per_chunk 20000 --start_index {start_index} --end_index {end_index}\n')

                
            
            print(f"Created file: {filename}")






