#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <stdio.h>
//创建多个CUDA流

__global__ void kernel1(float* data)
{
	const int SIZE = 1024;
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < SIZE)
	{
		data[idx] += 1.0f;
	}
}

__global__ void kernel2(float* data) 
{
	const int SIZE = 1024;
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (data[idx] < SIZE)
	{
		data[idx] += 2.0f;
	}
}

int main()
{
	const int SIZE = 1024;
	const int NUM_STREAMS = 4;
	float* d_data[NUM_STREAMS];
	float* h_data = new float[SIZE];
	for (int i = 0; i < SIZE; ++i)
	{
		h_data[i] = static_cast<float>(i);
	}
	// 创建CUDA流
	cudaStream_t streams[NUM_STREAMS];
	for (int i = 0; i < NUM_STREAMS; ++i)
	{
		cudaStreamCreate(&streams[i]);
	}
	// 在每个流中分配和复制数据
	for (int i = 0; i < NUM_STREAMS; ++i)
	{
		cudaMalloc(&d_data[i], SIZE * sizeof(float));
		cudaMemcpyAsync(d_data[i], h_data, SIZE * sizeof(float), cudaMemcpyHostToDevice, streams[i]);
	}

	// 启动核函数
	for (int i = 0; i < NUM_STREAMS; ++i)
	{
		if (i % 2 == 0)
		{
			kernel1 << <(SIZE + 256) / 256, 256, 0, streams[i] >> > (d_data[i]);
		}
		else
		{
			kernel2 << <(SIZE + 255) / 256, 256, 0, streams[i] >> > (d_data[i]);
		}
	}
	// 从设备复制数据到主机
	for (int i = 0; i < NUM_STREAMS; ++i)
	{
		cudaMemcpyAsync(h_data, d_data, SIZE * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
	}
	for (int i = 0; i < NUM_STREAMS; ++i)
	{
		cudaStreamSynchronize(streams[i]);
	}
	// 输出结果的部分数据
	for (int i = 0; i < 10; ++i) {
		std::cout << h_data[i] << " ";
	}
	std::cout << std::endl;

	// 释放资源
	for (int i = 0; i < NUM_STREAMS; ++i) {
		cudaFree(d_data[i]);
		cudaStreamDestroy(streams[i]);
	}

	delete[] h_data;

	return 0;
}


