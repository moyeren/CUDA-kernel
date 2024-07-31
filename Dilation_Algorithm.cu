#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <math.h>
#include <opencv2/opencv.hpp>

// 引入上述 CUDA 膨胀算法
__global__ void dilateKernel(const uchar* input, uchar* output, int width, int height, const int* kernel, int ksize);

void dilate(const uchar* input, uchar* output, int width, int height, int ksize);

int main() {
    // 图像尺寸
    const int width = 256;
    const int height = 256;
    const int ksize = 50;  // 结构元素尺寸

    // 创建一个简单的二值图像
    cv::Mat image(height, width, CV_8UC1, cv::Scalar(0));
    cv::rectangle(image, cv::Point(50, 50), cv::Point(200, 200), cv::Scalar(255), -1);

    // 将图像数据从 cv::Mat 转换为 uchar 数组
    uchar* host_input = image.data;
    uchar* host_output = new uchar[width * height];

    // 调用 CUDA 膨胀操作
    dilate(host_input, host_output, width, height, ksize);

    // 将输出数据转换为 cv::Mat
    cv::Mat output_image(height, width, CV_8UC1, host_output);

    // 显示图像
    cv::imshow("Original Image", image);
    cv::imshow("Dilated Image", output_image);
    cv::waitKey(0);

    // 保存图像
    cv::imwrite("dilated_image.png", output_image);

    // 清理
    delete[] host_output;

    return 0;
}

// 引入 CUDA 膨胀函数的实现
__global__ void dilateKernel(const uchar* input, uchar* output, int width, int height, const int* kernel, int ksize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int half_k = ksize / 2;
        bool has_foreground = false;

        for (int dy = -half_k; dy <= half_k; ++dy) {
            for (int dx = -half_k; dx <= half_k; ++dx) {
                int nx = x + dx;
                int ny = y + dy;
                if (nx >= 0 && ny >= 0 && nx < width && ny < height) {
                    int idx = ny * width + nx;
                    if (input[idx] > 0) {
                        has_foreground = true;
                        break;
                    }
                }
            }
            if (has_foreground) break;
        }

        output[y * width + x] = has_foreground ? 255 : 0;
    }
}

void dilate(const uchar* input, uchar* output, int width, int height, int ksize) {
    uchar* dev_input;
    uchar* dev_output;
    int* dev_kernel;

    size_t imageSize = width * height * sizeof(uchar);
    size_t kernelSize = ksize * ksize * sizeof(int);

    cudaMalloc(&dev_input, imageSize);
    cudaMalloc(&dev_output, imageSize);
    cudaMalloc(&dev_kernel, kernelSize);
    cudaMemcpy(dev_input, input, imageSize, cudaMemcpyHostToDevice);

    int* kernel = new int[ksize * ksize];
    for (int i = 0; i < ksize * ksize; ++i) {
        kernel[i] = 1;  // 全部设置为 1
    }
    cudaMemcpy(dev_kernel, kernel, kernelSize, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    dilateKernel << <gridSize, blockSize >> > (dev_input, dev_output, width, height, dev_kernel, ksize);

    cudaMemcpy(output, dev_output, imageSize, cudaMemcpyDeviceToHost);

    cudaFree(dev_input);
    cudaFree(dev_output);
    cudaFree(dev_kernel);
    delete[] kernel;
}
