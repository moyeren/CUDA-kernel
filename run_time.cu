#include <iostream>
#include <cuda_runtime.h>

__global__ void kernel() {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // 假设这里有一些计算
    for (int i = 0; i < 1000; ++i) {
        idx *= 2;
    }
}

int main() {
    // 定义并创建CUDA事件
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 预热运行
    kernel << <256, 256 >> > ();
    cudaDeviceSynchronize();

    // 记录起始事件
    cudaEventRecord(start, 0);

    // 调用CUDA核函数
    kernel << <256, 256 >> > ();

    // 记录结束事件
    cudaEventRecord(stop, 0);

    // 等待事件完成
    cudaEventSynchronize(stop);

    // 计算时间差
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    // 打印执行时间
    std::cout << "Elapsed time: " << elapsedTime << " ms" << std::endl;

    // 销毁CUDA事件
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
