#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

// CUDA 核函数：进行双线性插值
__global__ void bilinearInterpolationKernel(const unsigned char* input_image, unsigned char* output_image, int input_width, int input_height, int output_width, int output_height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < output_width && y < output_height) {
        // 计算输入图像中的坐标
        float fx = (float)x * (float)input_width / (float)output_width;
        float fy = (float)y * (float)input_height / (float)output_height;

        int x1 = (int)fx;
        int y1 = (int)fy;
        int x2 = min(x1 + 1, input_width - 1);
        int y2 = min(y1 + 1, input_height - 1);

        float dx = fx - x1;
        float dy = fy - y1;

        // 计算四个邻近像素的加权平均值
        unsigned char A = input_image[y1 * input_width + x1];
        unsigned char B = input_image[y1 * input_width + x2];
        unsigned char C = input_image[y2 * input_width + x1];
        unsigned char D = input_image[y2 * input_width + x2];

        float result = (1 - dx) * (1 - dy) * A +
            dx * (1 - dy) * B +
            (1 - dx) * dy * C +
            dx * dy * D;

        output_image[y * output_width + x] = static_cast<unsigned char>(result);
    }
}

int main() {
    // 利用 OpenCV 接口读取图片
    cv::Mat img = cv::imread("C:/figure/xiunv1.jpg", cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cout << "Could not open or find the image!" << std::endl;
        return -1;
    }

    int input_width = img.cols;
    int input_height = img.rows;
    int output_width = 1160; // 设定输出图像的宽度
    int output_height = 1160; // 设定输出图像的高度

    // 分配主机和设备内存
    unsigned char* h_input_image, * h_output_image;
    unsigned char* d_input_image, * d_output_image;

    size_t input_size = input_width * input_height * sizeof(unsigned char);
    size_t output_size = output_width * output_height * sizeof(unsigned char);

    h_input_image = (unsigned char*)malloc(input_size);
    h_output_image = (unsigned char*)malloc(output_size);

    cudaMalloc((void**)&d_input_image, input_size);
    cudaMalloc((void**)&d_output_image, output_size);

    // 将输入图像数据从 OpenCV 复制到主机内存
    memcpy(h_input_image, img.data, input_size);

    // 将输入图像数据从主机复制到设备
    cudaMemcpy(d_input_image, h_input_image, input_size, cudaMemcpyHostToDevice);

    // 配置 CUDA 网格和块
    dim3 block_size(16, 16);
    dim3 grid_size((output_width + block_size.x - 1) / block_size.x, (output_height + block_size.y - 1) / block_size.y);
    bilinearInterpolationKernel << <grid_size, block_size >> > (d_input_image, d_output_image, input_width, input_height, output_width, output_height);

    // 检查 CUDA 核函数调用后的错误
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        cout << "CUDA error: " << cudaGetErrorString(error) << endl;
        return -1;
    }

    // 将结果从设备复制到主机
    cudaMemcpy(h_output_image, d_output_image, output_size, cudaMemcpyDeviceToHost);

    // 将处理后的图像数据封装到 cv::Mat 中
    cv::Mat output_image(output_height, output_width, CV_8UC1, h_output_image);

    // 显示处理后的图像
    cv::imshow("Processed Image", output_image);
    cv::waitKey(0);

    // 释放内存
    free(h_input_image);
    free(h_output_image);
    cudaFree(d_input_image);
    cudaFree(d_output_image);

    return 0;
}
