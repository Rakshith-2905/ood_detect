#!/bin/bash

# Starting index
start_idx=0
# End index
end_idx=1000
# Increment
increment=100

while [ $start_idx -le $end_idx ]
do
    python generate_concept_set.py --start_idx=$start_idx --end_idx=$((start_idx + increment)) &
    start_idx=$((start_idx + increment))
done

wait # This waits for all background jobs to finish
