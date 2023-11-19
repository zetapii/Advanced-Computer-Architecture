import os
import multiprocessing

cores = multiprocessing.cpu_count() 
frequency = 3.2  
flops_per_cycle_per_core = 8  #Assuming AVX2 support and 8 FLOPs per cycle per core.
peak_gflops = cores * frequency * flops_per_cycle_per_core
	
print(f"Number of Cores: {cores}")
print(f"CPU Frequency (GHz): {frequency}")
print(f"Estimated Peak GFLOPS: {peak_gflops:.2f}")