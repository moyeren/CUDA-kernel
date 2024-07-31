#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <opencv2/opencv.hpp>

//二值化图像
__global__ void Binary(unsigned char* in_img, unsigned char* out_img, int width, int height,int thresh)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int idx = y * width + x;
    if (x <= width && y < height)
    {
        in_img[idx] >= thresh ? out_img[idx] = 0 : out_img[idx] = 255.0;
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


    // 分配主机和设备内存
    unsigned char* in_gpu_image;
    unsigned char* out_gpu_image;

    size_t num = input_width * input_height * sizeof(unsigned char);

    cudaMalloc((void**)&in_gpu_image, num);
    cudaMalloc((void**)&out_gpu_image, num);

    //返回的结果
    cv::Mat dst_img(input_height, input_width, CV_8UC1, cv::Scalar(0));

    // 将输入图像数据从主机复制到设备
    cudaMemcpy(in_gpu_image, img.data, num, cudaMemcpyHostToDevice);

    //设置二值化阈值
    int thresh = 60;

    // 配置 CUDA 网格和块
    dim3 block_size(32, 32);
    dim3 grid_size((input_width + block_size.x - 1) / block_size.x, (input_height + block_size.y - 1) / block_size.y);
    Binary<<<grid_size, block_size >>> (in_gpu_image, out_gpu_image, input_width, input_height, thresh);

    // 将结果从设备复制到主机
    cudaMemcpy(dst_img.data, out_gpu_image, num, cudaMemcpyDeviceToHost);

    // 显示处理后的图像
    cv::imshow("Processed Image", dst_img);
    cv::waitKey(0);

    // 释放内存
    cudaFree(in_gpu_image);
    cudaFree(out_gpu_image);

    return 0;
}
