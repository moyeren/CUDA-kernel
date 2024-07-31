#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <opencv2/opencv.hpp>

class MatrixOperations {
public:
    void matrixMultiplication(float* A, float* B, float* C, int m, int n, int k);
};

// 矩阵乘法内核函数
__global__ void matrixMultiplicationKernel(float* A, float* B, float* C, int m, int n, int k)
{
    int row = threadIdx.x + blockDim.x * blockIdx.x;
    int col = threadIdx.y + blockDim.y * blockIdx.y;

    if (row < m && col < k)
    {
        float sum = 0.0;
        for (int i = 0; i < n; i++)
        {
            sum += A[row * n + i] * B[i * k + col];
        }
        C[row * k + col] = sum;
    }
}

void MatrixOperations::matrixMultiplication(float* A, float* B, float* C, int m, int n, int k)
{
    float* d_A, * d_B, * d_C;
    size_t sizeA = m * n * sizeof(float);
    size_t sizeB = n * k * sizeof(float);
    size_t sizeC = m * k * sizeof(float);

    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeB, cudaMemcpyHostToDevice);
    dim3 blockSize(16, 16);
    dim3 gridSize((k + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y);
    matrixMultiplicationKernel << <gridSize, blockSize >> > (d_A, d_B, d_C, m, n, k);
    cudaMemcpy(C, d_C, sizeC, cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}



int main() {
    MatrixOperations matOps;

    // 矩阵乘法示例
    int m = 2, n = 3, k = 2;
    float A[] = { 1, 2, 3, 4, 5, 6 };
    float B[] = { 7, 8, 9, 10, 11, 12 };
    float* C = new float[m * k];

    matOps.matrixMultiplication(A, B, C, m, n, k);

    std::cout << "Matrix Multiplication Result:" << std::endl;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < k; ++j) {
            std::cout << C[i * k + j] << " ";
        }
        std::cout << std::endl;
    }

    delete[] C;

    return 0;
}
