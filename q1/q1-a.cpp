#include <iostream>
#include <vector>

using namespace std;

int dotProduct(const vector<int>& vector1, const vector<int>& vector2) 
{	
    int dot = 0;
    for (size_t i = 0; i < vector1.size(); i++) 
	{
        dot += vector1[i] * vector2[i];
    }
    return dot;
}

int main() 
{
    vector<int> vector_a = {1, 2, 3};
    vector<int> vector_b = {4, 5, 6};
 
    int result = dotProduct(vector_a, vector_b);
    cout << " dot product : " << result << endl;

    return 0;
}