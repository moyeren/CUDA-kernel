#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <opencv2/opencv.hpp>

//矩阵转置
__global__ void matrixTransposeKernel(float* input, float* output, int width, int height)
{
	__shared__ float tile[32][32];
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < width && y < height)
	{
		tile[threadIdx.y][threadIdx.x] = input[y * width + x];
	}
	__syncthreads();

	x = blockIdx.y * blockDim.x + threadIdx.x;
	y = blockIdx.x * blockDim.y + threadIdx.y;

	if (x < height && y < width)
	{
		output[y * height + x] = tile[threadIdx.x][threadIdx.y];
	}
}

void matrixTranspose(float* input,float* output,int width,int height)
{
	float* d_input, * d_output;
	size_t size = width * height * sizeof(float);
	cudaMalloc(&d_input, size);
	cudaMalloc(&d_output, size);

	cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);
	
	dim3 blockSize(32, 32);
	dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

	matrixTransposeKernel <<<gridSize, blockSize >>> (d_input, d_output, width, height);
	cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);
	cudaFree(d_input);
	cudaFree(d_output);
}

int main()
{
    int width = 10;
    int height = 5;
    float* input = new float[width * height];
    float* output = new float[width * height];

    // 初始化矩阵
    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            input[i * width + j] = i * width + j;
        }
    }

    // 打印输入矩阵
    std::cout << "Input Matrix:" << std::endl;
    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            std::cout << input[i * width + j] << " ";
        }
        std::cout << std::endl;
    }

    // 执行矩阵转置
    matrixTranspose(input, output, width, height);

    // 打印输出矩阵
    std::cout << "Output Transposed Matrix:" << std::endl;
    for (int i = 0; i < width; ++i)
    {
        for (int j = 0; j < height; ++j)
        {
            std::cout << output[i * height + j] << " ";
        }
        std::cout << std::endl;
    }

    delete[] input;
    delete[] output;

    return 0;
}
