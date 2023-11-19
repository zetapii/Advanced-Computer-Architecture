#include <iostream>
#include <vector>
#include <immintrin.h>

int dot_product_even_indices(const std::vector<int>& x, const std::vector<int>& y) 
{

    __m256i sum = _mm256_setzero_si256();

    for (size_t i = 0; i < x.size(); i += 8)
	{
        __m256i x_vec = _mm256_loadu_si256((__m256i*)&x[i]);
        __m256i y_vec = _mm256_loadu_si256((__m256i*)&y[i]);

        __m256i mask = _mm256_setr_epi32(0, -1, 0, -1, 0, -1, 0, -1);

        //   mask to filter out even indices
        x_vec = _mm256_and_si256(x_vec, mask);
        y_vec = _mm256_and_si256(y_vec, mask);

        __m256i prod = _mm256_mullo_epi32(x_vec, y_vec);
        sum = _mm256_add_epi32(sum, prod);
    }

    alignas(32) int result[8];
    _mm256_store_si256((__m256i*)result, sum);

    return result[0] + result[1] + result[2] + result[3] + result[4] + result[5] + result[6] + result[7];
}

int main() 
{
    std::vector<int> x = {1, 1, 1, 1, 1, 1, 1, 1}; 
    std::vector<int> y = {1, 2, 3, 4, 5, 6, 7, 8}; 

	int result = dot_product_even_indices(x, y);
    std::cout<<result;
    return 0;
}