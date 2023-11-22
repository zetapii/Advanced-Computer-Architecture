#include <iostream>
#include <vector>
using namespace std;

//normal matrix multiplication
void matrixMul(int n, int** matrixA, int** matrixB, int** matrixC) 
{
    int row = 0;
    int col = 0;
    int k = 0;
    for (row = 0; row < n; row++) 
    {
        for (col = 0; col < n; col++) 
        {
            for (k = 0; k < n; k++) 
            {
                matrixC[row][col] += matrixA[row][k] * matrixB[k][col];
            }
        }
    }
    return;
}


//strassen matrix multiplication
void add(vector<vector<int> >& A, vector<vector<int> >& B, vector<vector<int> >& C, int size) 
{
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            C[i][j] = A[i][j] + B[i][j];
}

void subtract(vector<vector<int> >& A, vector<vector<int> >& B, vector<vector<int> >& C, int size) 
{
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            C[i][j] = A[i][j] - B[i][j];
}

// Strassen's Matrix Multiplication
void strassen(vector< vector<int> >& A, vector< vector<int> >& B, vector< vector<int> >& C, int size) 
{
    if (size == 1) 
	{
        C[0][0] = A[0][0] * B[0][0];
        return;
    }

    int newSize = size / 2;
    vector<int> inner(newSize);
    vector<vector<int> > 
        A11(newSize, inner), A12(newSize, inner), A21(newSize, inner), A22(newSize, inner),
        B11(newSize, inner), B12(newSize, inner), B21(newSize, inner), B22(newSize, inner),
        C11(newSize, inner), C12(newSize, inner), C21(newSize, inner), C22(newSize, inner),
        M1(newSize, inner), M2(newSize, inner), M3(newSize, inner), M4(newSize, inner),
        M5(newSize, inner), M6(newSize, inner), M7(newSize, inner),
        AResult(newSize, inner), BResult(newSize, inner);

    for (int i = 0; i < newSize; i++)
	{
        for (int j = 0; j < newSize; j++) 
		{
            A11[i][j] = A[i][j];
            A12[i][j] = A[i][j + newSize];
            A21[i][j] = A[i + newSize][j];
            A22[i][j] = A[i + newSize][j + newSize];

            B11[i][j] = B[i][j];
            B12[i][j] = B[i][j + newSize];
            B21[i][j] = B[i + newSize][j];
            B22[i][j] = B[i + newSize][j + newSize];
        }
    }

    // Calculating M1 to M7
    add(A11, A22, AResult, newSize);
    add(B11, B22, BResult, newSize);
    strassen(AResult, BResult, M1, newSize);

    add(A21, A22, AResult, newSize);
    strassen(AResult, B11, M2, newSize);

    subtract(B12, B22, BResult, newSize);
    strassen(A11, BResult, M3, newSize);

    subtract(B21, B11, BResult, newSize);
    strassen(A22, BResult, M4, newSize);

    add(A11, A12, AResult, newSize);
    strassen(AResult, B22, M5, newSize);

    subtract(A21, A11, AResult, newSize);
    add(B11, B12, BResult, newSize);
    strassen(AResult, BResult, M6, newSize);

    subtract(A12, A22, AResult, newSize);
    add(B21, B22, BResult, newSize);
    strassen(AResult, BResult, M7, newSize);

    add(M1, M4, AResult, newSize);
    subtract(M7, M5, BResult, newSize);
    add(AResult, BResult, C11, newSize);

    add(M3, M5, C12, newSize);

    add(M2, M4, C21, newSize);

    add(M1, M3, AResult, newSize);
    subtract(M6, M2, BResult, newSize);
    add(AResult, BResult, C22, newSize);

    // Grouping the results into the final matrix
    for (int i = 0; i < newSize; i++)
	{
	        for (int j = 0; j < newSize; j++)
			{
                C[i][j] = C11[i][j];
                C[i][j + newSize] = C12[i][j];
                C[i + newSize][j] = C21[i][j];
                C[i + newSize][j + newSize] = C22[i][j];
	        }
    }
}

int main() 
{
    int n = (1<<8);
    
    int** matrixA = new int*[n];
    int** matrixB = new int*[n];
    int** matrixC = new int*[n];
    
    for (int i = 0; i < n; ++i) 
    {
        matrixA[i] = new int[n];
        matrixB[i] = new int[n];
        matrixC[i] = new int[n];
        
    }

    matrixMul(n, matrixA, matrixB, matrixC);

    for (int i = 0; i < n; ++i) 
    {
        delete[] matrixA[i];
        delete[] matrixB[i];
        delete[] matrixC[i];
    }

    delete[] matrixA;
    delete[] matrixB;
    delete[] matrixC;

    return 0;
}