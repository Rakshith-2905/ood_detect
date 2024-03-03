domain_names=("autumn") #"dim" "grass" "outdoor" "rock" "water" "outdoor") #"dim" "grass" "outdoor" "rock" "water")
agg=("mean" "max") #"mean" 
for a in "${!domain_names[@]}"; do
    for b in "${agg[@]}"; do
        sbatch NICOpp_PIM_train.sh ${domain_names[$a]} $b
done
done
