import os
out_dir="/g/g92/thopalli/ws_kdml/KDML/ood_detect/feat_extract_bash_files/feat_logs/dino_vits16/train"
out_files=os.listdir(out_dir)
out_files.sort()
completed_files=[]
for out_file in out_files:
    out_file_path=os.path.join(out_dir,out_file)
    with open(out_file_path,"r") as f:
        lines=f.readlines()
        for line in lines:
            if "completed" in line:
                completed_files.append(out_file)
                break


# print not completed files
for out_file in out_files:
    if out_file not in completed_files:
        print(out_file)


                