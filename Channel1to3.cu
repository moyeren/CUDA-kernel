#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <chrono>
#include <math.h>
#include <opencv2/opencv.hpp>


// CUDA 核函数：将单通道图像转换为三通道图像
__global__ void convertToThreeChannelKernel(unsigned char* d_input, unsigned char* d_output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        unsigned char pixel = d_input[idx];
        int outputIdx = (y * width + x) * 3;

        d_output[outputIdx] = pixel;       // Red channel
        d_output[outputIdx + 1] = pixel;   // Green channel
        d_output[outputIdx + 2] = pixel;   // Blue channel
    }
}

void convertToThreeChannel(const cv::Mat& inputImage, cv::Mat& outputImage) {
    int width = inputImage.cols;
    int height = inputImage.rows;
    size_t size = width * height * sizeof(unsigned char);

    // Allocate device memory
    unsigned char* d_input;
    unsigned char* d_output;
    cudaError_t err;

    err = cudaMalloc(&d_input, size);
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc error for d_input: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    err = cudaMalloc(&d_output, width * height * 3 * sizeof(unsigned char)); // 3 channels
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc error for d_output: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input);
        return;
    }

    // Copy image data to device
    err = cudaMemcpy(d_input, inputImage.data, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy error for d_input: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        return;
    }

    // Launch the kernel
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);
    convertToThreeChannelKernel << <gridDim, blockDim >> > (d_input, d_output, width, height);

    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel error: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        return;
    }

    // Copy result back to host
    outputImage.create(height, width, CV_8UC3); // Three channels
    err = cudaMemcpy(outputImage.data, d_output, width * height * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy error for d_output: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        return;
    }

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

int main() {
    // Load a grayscale image
    cv::Mat inputImage = cv::imread("C:/figure/xiunv.jpg", cv::IMREAD_GRAYSCALE);
    if (inputImage.empty()) {
        std::cerr << "Error loading image!" << std::endl;
        return -1;
    }

    // Display the grayscale image
    cv::imshow("Original Image", inputImage);
    cv::waitKey(0); // Wait for a key press

    // Prepare output image
    cv::Mat outputImage;

    // Convert to three-channel image
    convertToThreeChannel(inputImage, outputImage);

    // Save the result
    cv::imwrite("output_image.png", outputImage);

    // Display the converted image
    cv::imshow("Converted Image", outputImage);
    cv::waitKey(0); // Wait for a key press



    return 0;
}




