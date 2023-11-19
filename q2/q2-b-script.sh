#!/bin/bash

PROGRAM="./a.out"

L1_EVENTS="L1-dcache-loads,L1-dcache-load-misses"
L2_EVENTS="l2_rqsts.all_demand_references,l2_rqsts.all_demand_miss"
L3_EVENTS="LLC-loads,LLC-load-misses,LLC-stores,LLC-store-misses"
CPI="instructions,cycles,cache-references,cache-misses"

run_perf() {
  local events="$1"
  local size="$2"
  echo "Matrix Size: $size" >> tmp.txt
  perf stat -e "$events" "$PROGRAM" "$size" &>> tmp.txt
}

for (( k = 4; k <= 9; k++ )); do
  size=$((2**k))

  echo "Profiling L1 Cache for matrix size $size:" >> tmp.txt
  run_perf "$L1_EVENTS" "$size"

  echo "Profiling L2 Cache for matrix size $size:" >> tmp.txt
  run_perf "$L2_EVENTS" "$size"

  echo "Profiling L3 Cache for matrix size $size:" >> tmp.txt
  run_perf "$L3_EVENTS" "$size"
  
  run_perf "$CPI" "$size"

done
