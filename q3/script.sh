#!/bin/bash

SOURCE_FILE="/home/aabid/DSA/temp.cpp"

COMMANDS=(
  "valgrind --tool=cachegrind --I1=8192,128,64 --D1=8192,128,64 --L2=16384,256,64 ./a.out"
  "valgrind --tool=cachegrind --I1=16384,256,64 --D1=16384,256,64 --L2=32768,512,64 ./a.out"
  "valgrind --tool=cachegrind --I1=32768,512,64 --D1=32768,512,64 --L2=65536,1024,64 ./a.out"
  "valgrind --tool=cachegrind --I1=65536,1024,64 --D1=65536,1024,64 --L2=131072,2048,64 ./a.out"
  "valgrind --tool=cachegrind --I1=131072,2048,64 --D1=131072,2048,64 --L2=262144,4096,64 ./a.out"
  "valgrind --tool=cachegrind --I1=262144,4096,64 --D1=262144,4096,64 --L2=524288,8192,64 ./a.out"
  "valgrind --tool=cachegrind --I1=524288,8192,64 --D1=524288,8192,64 --L2=1048576,16384,64 ./a.out"
  "valgrind --tool=cachegrind --I1=1048576,16384,64 --D1=1048576,16384,64 --L2=2097152,32768,64 ./a.out"
)

for i in "${!COMMANDS[@]}"; do
  echo "Running configuration $((i + 1))..."
  eval "${COMMANDS[i]}"
  
  LAST_FILE=$(ls -t cachegrind.out.* | head -n 1)
  
  if [ -f "$LAST_FILE" ]; then
    echo "Annotating $LAST_FILE..."
    cg_annotate "$LAST_FILE" "$SOURCE_FILE" >> "tmp_$((i + 1)).txt"
  else
    echo "Cachegrind output file not found for configuration $((i + 1))"
  fi
done

echo "All configurations processed."