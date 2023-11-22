

#include <iostream>
#include <vector>
using namespace std;

//normal matrix multiplication

void mulBlocking(int n, int** matrixA, int** matrixB, int** matrixC,int BLOCK_SIZE)
{
    int block_size = BLOCK_SIZE; 
    int row_start = 0;
    int col_start = 0;
    int k_start = 0;
    int row = 0;
    int col = 0;
    int k = 0;
    for(row_start = 0; row_start < n; row_start += block_size)
    {
        for(col_start = 0; col_start < n; col_start += block_size)
        {
            for(k_start = 0; k_start < n; k_start += block_size)
            {
                for(row = 0; row < block_size; row++)
                {
                    for(col = 0; col < block_size; col++)
                    {
                        for(k = 0; k < block_size; k++)
                        {
                            matrixC[row_start + row][col_start + col] += 
                                matrixA[row_start + row][k_start + k] * 
                                matrixB[k_start + k][col_start + col];
                        }
                    }
                }
            }
        }
    }
    return ;
}

int main(int argc,char* argv[]) 
{
    int n = 512;
    
    int** matrixA = new int*[n];
    int** matrixB = new int*[n];
    int** matrixC = new int*[n];
    
    for (int i = 0; i < n; ++i) 
    {
        matrixA[i] = new int[n];
        matrixB[i] = new int[n];
        matrixC[i] = new int[n];
        
    }

    mulBlocking(n, matrixA, matrixB, matrixC,atoi(argv[1]));

    return 0;
}