#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>

#define BLOCK_SIZE 16

//相关系数实现模板匹配

// CUDA内核函数: 计算相关系数
__global__ void matchTemplateKernel(const unsigned char* d_image, const unsigned char* d_template,
    float* d_result, int imageWidth, int imageHeight,
    int templateWidth, int templateHeight) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < imageWidth - templateWidth + 1 && y < imageHeight - templateHeight + 1) {
        float sum_image = 0.0f, sum_template = 0.0f;
        float sum_image_sq = 0.0f, sum_template_sq = 0.0f, sum_product = 0.0f;

        for (int j = 0; j < templateHeight; ++j) {
            for (int i = 0; i < templateWidth; ++i) {
                int imgIndex = (y + j) * imageWidth + (x + i);
                int tmplIndex = j * templateWidth + i;

                float img_val = static_cast<float>(d_image[imgIndex]);
                float tmpl_val = static_cast<float>(d_template[tmplIndex]);

                sum_image += img_val;
                sum_template += tmpl_val;
                sum_image_sq += img_val * img_val;
                sum_template_sq += tmpl_val * tmpl_val;
                sum_product += img_val * tmpl_val;
            }
        }

        float num = sum_product - (sum_image * sum_template / (templateWidth * templateHeight));
        float den = sqrt((sum_image_sq - (sum_image * sum_image / (templateWidth * templateHeight))) *
            (sum_template_sq - (sum_template * sum_template / (templateWidth * templateHeight))));

        if (den == 0.0f) {
            d_result[y * (imageWidth - templateWidth + 1) + x] = 0.0f; // Avoid division by zero
        }
        else {
            d_result[y * (imageWidth - templateWidth + 1) + x] = num / den;
        }
    }
}

int main() {
    // Load images using OpenCV
    cv::Mat image = cv::imread("C:/figure/xiunv1.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat templateImage = cv::imread("C:/figure/xiunv1_template.jpg", cv::IMREAD_GRAYSCALE);

    if (image.empty() || templateImage.empty()) {
        std::cerr << "Error loading images!" << std::endl;
        return -1;
    }

    int imageWidth = image.cols;
    int imageHeight = image.rows;
    int templateWidth = templateImage.cols;
    int templateHeight = templateImage.rows;

    unsigned char* d_image;
    unsigned char* d_template;
    float* d_result;

    size_t imageSize = imageWidth * imageHeight * sizeof(unsigned char);
    size_t templateSize = templateWidth * templateHeight * sizeof(unsigned char);
    size_t resultSize = (imageWidth - templateWidth + 1) * (imageHeight - templateHeight + 1) * sizeof(float);

    // Allocate device memory
    cudaMalloc(&d_image, imageSize);
    cudaMalloc(&d_template, templateSize);
    cudaMalloc(&d_result, resultSize);

    // Copy data from host to device
    cudaMemcpy(d_image, image.data, imageSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_template, templateImage.data, templateSize, cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((imageWidth - templateWidth + 1 + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (imageHeight - templateHeight + 1 + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch kernel
    matchTemplateKernel << <gridSize, blockSize >> > (d_image, d_template, d_result,
        imageWidth, imageHeight,
        templateWidth, templateHeight);
    cudaDeviceSynchronize();

    // Allocate host memory for result
    float* h_result = new float[(imageWidth - templateWidth + 1) * (imageHeight - templateHeight + 1)];

    // Copy result from device to host
    cudaMemcpy(h_result, d_result, resultSize, cudaMemcpyDeviceToHost);

    // Find the best match position
    float maxScore = -FLT_MAX;
    int bestX = 0;
    int bestY = 0;
    for (int y = 0; y < imageHeight - templateHeight + 1; ++y) {
        for (int x = 0; x < imageWidth - templateWidth + 1; ++x) {
            float score = h_result[y * (imageWidth - templateWidth + 1) + x];
            if (score > maxScore) {
                maxScore = score;
                bestX = x;
                bestY = y;
            }
        }
    }

    std::cout << "Best match found at: (" << bestX << ", " << bestY << ")" << std::endl;

    // Clean up
    cudaFree(d_image);
    cudaFree(d_template);
    cudaFree(d_result);
    delete[] h_result;

    return 0;
}
