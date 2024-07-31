#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

//旋转和错切输入图像

//输入图像input、输出图像output、旋转中心center_x和center_y、旋转角度theta以及错切参数shear。
__global__ void rotateKernel(uchar* input, uchar* output, int center_x, int center_y, int width, int height, float theta, float shear) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        size_t current_idx = static_cast<size_t>(y) * width + x;

        // 计算旋转后的坐标点
        float sin_theta = sin(theta);
        float cos_theta = cos(theta);
        float x_prime = (cos_theta - shear * sin_theta) * (x - center_x) + (shear * cos_theta + sin_theta) * (y - center_y) + center_x;
        float y_prime = (-sin_theta * (x - center_x) + cos_theta * (y - center_y) + center_y);

        // 使用双线性插值计算输出像素值
        if (x_prime < 0 || x_prime >= width - 1 || y_prime < 0 || y_prime >= height - 1) {
            output[current_idx] = 0;
        }
        else {
            int x1 = floor(x_prime);
            int y1 = floor(y_prime);
            float dx = x_prime - x1;
            float dy = y_prime - y1;

            float pixel_lt = input[y1 * width + x1];
            float pixel_rt = input[y1 * width + x1 + 1];
            float pixel_lb = input[(y1 + 1) * width + x1];
            float pixel_rb = input[(y1 + 1) * width + x1 + 1];

            output[current_idx] = (1 - dx) * (1 - dy) * pixel_lt + dx * (1 - dy) * pixel_rt + (1 - dx) * dy * pixel_lb + dx * dy * pixel_rb;
        }
    }
}

int main() {
    // Load images using OpenCV
    cv::Mat image = cv::imread("C:/figure/xiunv1.jpg", cv::IMREAD_GRAYSCALE);

    if (image.empty()) {
        std::cerr << "Error loading images!" << std::endl;
        return -1;
    }

    int imageWidth = image.cols;
    int imageHeight = image.rows;
    int BLOCK_SIZE = 32;

    unsigned char* d_input;
    unsigned char* d_output;
    

    //!旋转错切的大小以及中心
    int center_x = imageWidth / 2;
    int center_y = imageHeight / 2;
    float theta = 30 * M_PI / 180.0f; // 将角度转换为弧度
    float shear = 10 * M_PI / 180.0f; // 将角度转换为弧度

    size_t imageSize = imageWidth * imageHeight * sizeof(unsigned char);

    // Allocate device memory
    cudaMalloc(&d_input, imageSize);
    cudaMalloc(&d_output, imageSize);

    // Copy data from host to device
    cudaMemcpy(d_input, image.data, imageSize, cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((imageWidth + BLOCK_SIZE - 1) / BLOCK_SIZE, (imageHeight + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch kernel
    rotateKernel << <gridSize, blockSize >> > (d_input, d_output, center_x, center_y, imageWidth, imageHeight, theta, shear);
    cudaDeviceSynchronize();

    // Allocate host memory for result
    cv::Mat h_result(imageHeight, imageWidth, CV_8UC1, cv::Scalar(0));

    // Copy result from device to host
    cudaMemcpy(h_result.data, d_output, imageSize, cudaMemcpyDeviceToHost);

    cv::imshow("RESULT", h_result);
    cv::waitKey(0);

    // Clean up
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
