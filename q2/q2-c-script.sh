#!/bin/bash

SOURCE_FILE="/home/zaid/DSA/temp.cpp"

BLOCK_SIZES=(8 16 32 64 128)

L1_SIZE=32768  # 32 KB
L2_SIZE=1048576  # 1 MB
LINE_SIZE=64  # 64 bytes
L1_LINES=$((L1_SIZE / LINE_SIZE))
L2_LINES=$((L2_SIZE / LINE_SIZE))

# Loop through each block size
for BLOCK_SIZE in "${BLOCK_SIZES[@]}"; do
  echo "Running with block size: $BLOCK_SIZE..."

  valgrind  --tool=cachegrind --I1=$L1_SIZE,$L1_LINES,$LINE_SIZE --D1=$L1_SIZE,$L1_LINES,$LINE_SIZE --L2=$L2_SIZE,$L2_LINES,$LINE_SIZE ./a.out "$BLOCK_SIZE"

  LAST_FILE=$(ls -t cachegrind.out.* | head -n 1)
  
  if [ -f "$LAST_FILE" ]; then
    echo "Annotating $LAST_FILE for block size $BLOCK_SIZE..."
    cg_annotate "$LAST_FILE" "$SOURCE_FILE" >> "tmp_block_size_$BLOCK_SIZE.txt"
  else
    echo "Cachegrind output file not found for block size $BLOCK_SIZE"
  fi
done

echo "All block sizes processed."