#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

__global__ void getHistKernelV3(const uchar* input, unsigned int* histogram, int cols, int rows)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < cols && y < rows) {
        // 计算当前像素在一维数据中的索引
        size_t current_idx = static_cast<size_t>(y) * cols + x;
        uchar pixelValue = input[current_idx]; // 读取当前像素值

        // 将当前像素值对应的直方图bin的计数加一
        atomicAdd(&histogram[pixelValue], 1);
    }
}

int main()
{
    // 利用Opencv接口读取图片
    Mat img = imread("C:/figure/xiunv1.jpg", IMREAD_GRAYSCALE);
    if (img.empty()) {
        cout << "Could not open or find the image!" << endl;
        return -1;
    }
    int imgWidth = img.cols;
    int imgHeight = img.rows;

    // 利用Opencv对读入的grayimg进行去噪
    Mat gaussImg;
    GaussianBlur(img, gaussImg, Size(3, 3), 0, 0, BORDER_DEFAULT);

    // 调用函数
    size_t num = imgHeight * imgWidth * sizeof(unsigned char);
    unsigned char* in_gpu;
    unsigned int* histogram_gpu;
    cudaMalloc((void**)&in_gpu, num);
    cudaMalloc((void**)&histogram_gpu, 256 * sizeof(unsigned int));
    cudaMemset(histogram_gpu, 0, 256 * sizeof(unsigned int)); // 初始化直方图为0

    dim3 threadPerBlock(32, 32);
    dim3 blocksPerGrid((imgWidth + threadPerBlock.x - 1) / threadPerBlock.x, (imgHeight + threadPerBlock.y - 1) / threadPerBlock.y);
    cudaMemcpy(in_gpu, gaussImg.data, num, cudaMemcpyHostToDevice);
    getHistKernelV3 << <blocksPerGrid, threadPerBlock >> > (in_gpu, histogram_gpu, imgWidth, imgHeight);

    unsigned int histogram[256] = { 0 };
    cudaMemcpy(histogram, histogram_gpu, 256 * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 256; i++) {
        cout << "Value " << i << ": " << histogram[i] << endl;
    }

    cudaFree(in_gpu);
    cudaFree(histogram_gpu);
    return 0;
}
