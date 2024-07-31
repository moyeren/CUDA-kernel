#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

// x0 x1 x2
// x3 x4 x5
// x6 x7 x8
__global__ void sobel_gpu(const unsigned char* in, unsigned char* out, int imgHeight, int imgWidth, int channels)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int index = (y * imgWidth + x) * channels;

    if (x > 0 && x < imgWidth - 1 && y > 0 && y < imgHeight - 1)
    {
        int Gx[3] = { 0, 0, 0 };  // Gradient in X direction for R, G, B
        int Gy[3] = { 0, 0, 0 };  // Gradient in Y direction for R, G, B

        // Define Sobel filters
        const int sobelX[3][3] = { {-1, 0, 1},
                                  {-2, 0, 2},
                                  {-1, 0, 1} };
        const int sobelY[3][3] = { {-1, -2, -1},
                                  {0, 0, 0},
                                  {1, 2, 1} };

        for (int c = 0; c < channels; c++) {
            unsigned char x0 = in[((y - 1) * imgWidth + (x - 1)) * channels + c];
            unsigned char x1 = in[((y - 1) * imgWidth + (x)) * channels + c];
            unsigned char x2 = in[((y - 1) * imgWidth + (x + 1)) * channels + c];
            unsigned char x3 = in[((y)*imgWidth + (x - 1)) * channels + c];
            unsigned char x4 = in[((y)*imgWidth + (x)) * channels + c];
            unsigned char x5 = in[((y)*imgWidth + (x + 1)) * channels + c];
            unsigned char x6 = in[((y + 1) * imgWidth + (x - 1)) * channels + c];
            unsigned char x7 = in[((y + 1) * imgWidth + (x)) * channels + c];
            unsigned char x8 = in[((y + 1) * imgWidth + (x + 1)) * channels + c];

            Gx[c] = (x2 + x5 * 2 + x8) - (x0 + x3 * 2 + x6);
            Gy[c] = (x0 + x1 * 2 + x2) - (x6 + x7 * 2 + x8);
        }

        for (int c = 0; c < channels; c++) {
            int magnitude = min(255, max(0, (abs(Gx[c]) + abs(Gy[c])) / 2));
            out[index + c] = static_cast<unsigned char>(magnitude);
        }
    }
}

int main()
{
    // 利用Opencv接口读取彩色图片
    Mat img = imread("C:/figure/xiunv1.jpg", IMREAD_COLOR);
    if (img.empty()) {
        cout << "Could not open or find the image!" << endl;
        return -1;
    }
    int imgWidth = img.cols;
    int imgHeight = img.rows;
    int channels = img.channels();

    // 利用Opencv对读入的图像进行去噪
    Mat gaussImg;
    GaussianBlur(img, gaussImg, Size(3, 3), 0, 0, BORDER_DEFAULT);

    // GPU结果为：dst_gpu
    Mat dst_gpu(imgHeight, imgWidth, CV_8UC3, Scalar(0, 0, 0));

    // 调用函数
    size_t num = imgHeight * imgWidth * channels * sizeof(unsigned char);
    unsigned char* in_gpu;
    unsigned char* out_gpu;
    cudaMalloc((void**)&in_gpu, num);
    cudaMalloc((void**)&out_gpu, num);

    dim3 threadPerBlock(32, 32);
    dim3 blocksPerGrid((imgWidth + threadPerBlock.x - 1) / threadPerBlock.x, (imgHeight + threadPerBlock.y - 1) / threadPerBlock.y);
    cudaMemcpy(in_gpu, gaussImg.data, num, cudaMemcpyHostToDevice);
    sobel_gpu << <blocksPerGrid, threadPerBlock >> > (in_gpu, out_gpu, imgHeight, imgWidth, channels);
    cudaMemcpy(dst_gpu.data, out_gpu, num, cudaMemcpyDeviceToHost);

    imshow("GPU", dst_gpu);

    waitKey(0);

    cudaFree(in_gpu);
    cudaFree(out_gpu);
    return 0;
}
