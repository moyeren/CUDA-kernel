#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <opencv2/opencv.hpp>

__global__ void cropKernel(unsigned char* d_input, unsigned char* d_output, int width, int height, int x_start, int y_start, int crop_width, int crop_height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < crop_width && y < crop_height) {
        int input_x = x + x_start;
        int input_y = y + y_start;
        if (input_x < width && input_y < height) {
            int input_idx = input_y * width + input_x;
            int output_idx = y * crop_width + x;
            d_output[output_idx] = d_input[input_idx];
        }
    }
}

void cropImage(unsigned char* h_input, unsigned char* h_output, int width, int height, int x_start, int y_start, int crop_width, int crop_height)
{
    unsigned char* d_input, * d_output;

    size_t input_size = width * height * sizeof(unsigned char);
    size_t output_size = crop_width * crop_height * sizeof(unsigned char);

    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_output, output_size);

    cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((crop_width + blockDim.x - 1) / blockDim.x, (crop_height + blockDim.y - 1) / blockDim.y);

    cropKernel << <gridDim, blockDim >> > (d_input, d_output, width, height, x_start, y_start, crop_width, crop_height);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        exit(-1);
    }

    cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}

int main() {
    cv::Mat img = cv::imread("C:/figure/xiunv1.jpg", cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cout << "Could not open or find the image!" << std::endl;
        return -1;
    }

    int width = img.cols;
    int height = img.rows;
    const int x_start = 1000;
    const int y_start = 500;
    const int crop_width = 300;
    const int crop_height = 300;

    unsigned char* h_output = new unsigned char[crop_width * crop_height];
    unsigned char* h_input = img.data; // Use OpenCV Mat data directly

    cropImage(h_input, h_output, width, height, x_start, y_start, crop_width, crop_height);

    // Create a cv::Mat from the cropped data
    cv::Mat h_output_new(crop_height, crop_width, CV_8UC1, h_output);

    // Display the cropped image
    cv::imshow("CROP_PIC", h_output_new);
    cv::waitKey(0);

    delete[] h_output; // Free the memory allocated for cropped image

    return 0;
}
