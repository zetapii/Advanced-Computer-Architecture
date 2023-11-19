#include <iostream>
#include <chrono>

const long long ARRAY_SIZE = 1000000000;
const int NUM_RUNS = 10;

void stream_copy(double* dest, const double* src, long long size) 
{
    for (long long i = 0; i < size; ++i) 
    {
        dest[i] = src[i];
    }
}

int main()
{
    double total_bandwidth = 0.0;
    for(int run = 0; run < NUM_RUNS; ++run) 
	{
        double* A = new double[ARRAY_SIZE];
        double* B = new double[ARRAY_SIZE];
        for(long long i = 0; i < ARRAY_SIZE; ++i) 
        {
            A[i] = 1.0;
            B[i] = 2.0;
        }
        auto start = std::chrono::high_resolution_clock::now();
        //  memory-bound operation 
        stream_copy(A, B, ARRAY_SIZE);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        double bandwidth = (double(ARRAY_SIZE) * sizeof(double)) / (duration.count() * 1e9);
        total_bandwidth += bandwidth;
        delete[] A;
        delete[] B;
    }
    double average_bandwidth = total_bandwidth / NUM_RUNS;
    std::cout << "Average Memory Bandwidth: " << average_bandwidth << " GB/s" << std::endl;
    return 0;
}