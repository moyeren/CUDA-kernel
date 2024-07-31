#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

__global__ void affine_img(const float* affine, uchar* Output_image, const uchar* Input_image, int width, int height)
{
    int x = (blockDim.x * blockIdx.x) + threadIdx.x;
    int y = (blockDim.y * blockIdx.y) + threadIdx.y;

    if (x < width && y < height) {
        int global_idx = y * width + x;

        float matrix0 = affine[0];
        float matrix1 = affine[1];
        float matrix2 = affine[2];
        float matrix3 = affine[3];
        float matrix4 = affine[4];
        float matrix5 = affine[5];

        float proj_x = matrix0 * x + matrix1 * y + matrix2;
        float proj_y = matrix3 * x + matrix4 * y + matrix5;

        if (proj_x >= 0 && proj_x < width && proj_y >= 0 && proj_y < height) {
            int proj_idx = static_cast<int>(proj_y) * width + static_cast<int>(proj_x);
            Output_image[global_idx] = Input_image[proj_idx];
        }
        else {
            Output_image[global_idx] = 0; // 黑色像素
        }
    }
}

int main()
{
    // 读取图像
    Mat img = imread("C:/figure/xiunv1.jpg", IMREAD_GRAYSCALE);
    if (img.empty()) {
        cout << "Could not open or find the image!" << endl;
        return -1;
    }

    int imgWidth = img.cols;
    int imgHeight = img.rows;

    // 去噪
    Mat gaussImg;
    GaussianBlur(img, gaussImg, Size(3, 3), 0, 0, BORDER_DEFAULT);

    // GPU结果图像
    Mat dst_gpu(imgHeight, imgWidth, CV_8UC1, Scalar(0));

    // CUDA 内存分配
    size_t num = imgHeight * imgWidth * sizeof(unsigned char);
    const int matrix_size = 6 * sizeof(float);
    uchar* d_in_gpu;
    uchar* d_out_gpu;
    float* d_affine_matrix;

    cudaMalloc((void**)&d_in_gpu, num);
    cudaMalloc((void**)&d_out_gpu, num);
    cudaMalloc((void**)&d_affine_matrix, matrix_size);

    float h_matrix[6] = { 1.0f, 0.6f, 0.0f,
                         0.0f, 1.0f, 0.0f }; // 示例仿射矩阵

    // 将数据从主机复制到设备
    cudaMemcpy(d_in_gpu, gaussImg.data, num, cudaMemcpyHostToDevice);
    cudaMemcpy(d_affine_matrix, h_matrix, matrix_size, cudaMemcpyHostToDevice);

    // 配置 CUDA 内核
    dim3 threadPerBlock(32, 32);
    dim3 blocksPerGrid((imgWidth + threadPerBlock.x - 1) / threadPerBlock.x,
        (imgHeight + threadPerBlock.y - 1) / threadPerBlock.y);

    // 调用 CUDA 内核
    affine_img << <blocksPerGrid, threadPerBlock >> > (d_affine_matrix, d_out_gpu, d_in_gpu, imgWidth, imgHeight);

    // 从设备复制结果到主机
    cudaMemcpy(dst_gpu.data, d_out_gpu, num, cudaMemcpyDeviceToHost);

    // 显示结果图像
    imshow("Transformed Image", dst_gpu);
    waitKey(0);

    // 清理
    cudaFree(d_in_gpu);
    cudaFree(d_out_gpu);
    cudaFree(d_affine_matrix);

    return 0;
}
