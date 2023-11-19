#include <iostream>
#include <vector>
#include <immintrin.h>

int dot_product_vectorized(const std::vector<int> &x, const std::vector<int> &y)
{
    const size_t vectorSize = x.size();

    __m256i sum = _mm256_setzero_si256();

    for (size_t i = 0; i < vectorSize; i += 8)  // AVX2 registers can handle 8 integers at a time
    {
        __m256i x_vec = _mm256_loadu_si256((__m256i *)&x[i]);
        __m256i y_vec = _mm256_loadu_si256((__m256i *)&y[i]);

        __m256i prod = _mm256_mullo_epi32(x_vec, y_vec);
        sum = _mm256_add_epi32(sum, prod);
    }

    alignas(32) int result[8];
    _mm256_store_si256((__m256i *)result, sum);

    int finalResult = 0;
    for (int i = 0; i < 8; ++i) 
        finalResult += result[i];

    return finalResult;
}

int main()
{
    std::vector<int> x = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<int> y = {2, 3, 4, 5, 6, 7, 8, 9};

    int result = dot_product_vectorized(x, y);
    std::cout<< result << std::endl;
    return 0;
}