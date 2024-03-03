train_domain_names=("autumn" "dim" "grass" "outdoor" "rock" "water") #"dim" "grass" "outdoor" "rock" "water" "outdoor") #"dim" "grass" "outdoor" "rock" "water")
agg=("mean" "max") #"mean" 
methods=("pim")
for a in "${!train_domain_names[@]}"; do
    for b in "${!agg[@]}"; do
        for c in "${methods[@]}"; do
            sbatch NICOpp_PIM_failure_eval.sh ${train_domain_names[$a]} ${agg[$b]} $c
done
done
done
