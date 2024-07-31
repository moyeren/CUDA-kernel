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
__global__ void sobel_gpu(unsigned char* in, unsigned char* out, int imgHeight, int imgWidth)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int index = y * imgWidth + x;
    int Gx = 0;
    int Gy = 0;
    unsigned char x0, x1, x2, x3, x4, x5, x6, x7, x8;
    if (x > 0 && x < imgWidth && y > 0 && y < imgHeight)
    {
        x0 = in[(y - 1) * imgWidth + (x - 1)];
        x1 = in[(y - 1) * imgWidth + (x)];
        x2 = in[(y - 1) * imgWidth + (x + 1)];
        x3 = in[(y)*imgWidth + (x - 1)];
        x4 = in[(y)*imgWidth + (x)];
        x5 = in[(y)*imgWidth + (x + 1)];
        x6 = in[(y + 1) * imgWidth + (x - 1)];
        x7 = in[(y + 1) * imgWidth + (x)];
        x8 = in[(y + 1) * imgWidth + (x + 1)];
        Gx = (x2 + x5 * 2 + x8) - (x0 + x3 * 2 + x6);
        Gy = (x0 + x1 * 2 + x2) - (x6 + x7 * 2 + x8);
        out[index] = (abs(Gx) + abs(Gy)) / 2;
    }
}

int main()
{
    //利用Opencv接口读取图片
    Mat img = imread("C:/figure/xiunv1.jpg", IMREAD_GRAYSCALE);
    if (img.empty()) {
        cout << "Could not open or find the image!" << endl;
        return -1;
    }
    int imgWidth = img.cols;
    int imgHeight = img.rows;

    //利用Opencv对读入的grayimg进行去噪
    Mat gaussImg;
    GaussianBlur(img, gaussImg, Size(3, 3), 0, 0, BORDER_DEFAULT);

    //GPU结果为：dst_gpu
    Mat dst_gpu(imgHeight, imgWidth, CV_8UC1, Scalar(0));

    //调用函数
    size_t num = imgHeight * imgWidth * sizeof(unsigned char);
    unsigned char* in_gpu;
    unsigned char* out_gpu;
    cudaMalloc((void**)&in_gpu, num);
    cudaMalloc((void**)&out_gpu, num);

    dim3 threadPerBlock(32, 32);
    dim3 blocksPerGrid((imgWidth + threadPerBlock.x - 1) / threadPerBlock.x, (imgHeight + threadPerBlock.y - 1) / threadPerBlock.y);
    cudaMemcpy(in_gpu, gaussImg.data, num, cudaMemcpyHostToDevice);
    sobel_gpu << <blocksPerGrid, threadPerBlock >> > (in_gpu, out_gpu, imgHeight, imgWidth);
    cudaMemcpy(dst_gpu.data, out_gpu, num, cudaMemcpyDeviceToHost);

    imshow("GPU", dst_gpu);

    waitKey(0);

    cudaFree(in_gpu);
    cudaFree(out_gpu);
    return 0;
}