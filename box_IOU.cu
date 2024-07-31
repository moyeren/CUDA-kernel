#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <math.h>
#include <opencv2/opencv.hpp>
//计算两个矩形框的Iou
static __device__ float boxIou(
    float aleft, float atop, float aright, float abottom,
    float bleft, float btop, float bright, float bbottom
) {
    float cleft = max(aleft, bleft);
    float ctop = max(atop, btop);
    float cright = min(aright, bright);
    float cbottom = min(abottom, bbottom);

    float c_area = max(cright - cleft, 0.0f) * max(cbottom - ctop, 0.0f);
    if (c_area == 0.0f)
        return 0.0f;

    float a_area = max(0.0f, aright - aleft) * max(0.0f, abottom - atop);
    float b_area = max(0.0f, bright - bleft) * max(0.0f, bbottom - btop);
    return c_area / (a_area + b_area - c_area);
}

__global__ void calculateIoU(
    float* aleft, float* atop, float* aright, float* abottom,
    float* bleft, float* btop, float* bright, float* bbottom,
    float* iou, int num_boxes)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_boxes)
    {
        iou[idx] = boxIou(
            aleft[idx], atop[idx], aright[idx], abottom[idx],
            bleft[idx], btop[idx], bright[idx], bbottom[idx]
        );
    }
}

int main()
{
    // Define the number of boxes
    int num_boxes = 5;

    // Host vectors
    std::vector<float> h_aleft(num_boxes, 0.0f);
    std::vector<float> h_atop(num_boxes, 0.0f);
    std::vector<float> h_aright(num_boxes, 1.0f);
    std::vector<float> h_abottom(num_boxes, 1.0f);

    std::vector<float> h_bleft(num_boxes, 0.5f);
    std::vector<float> h_btop(num_boxes, 0.5f);
    std::vector<float> h_bright(num_boxes, 1.5f);
    std::vector<float> h_bbottom(num_boxes, 1.5f);

    std::vector<float> h_iou(num_boxes);

    // Device vectors
    float* d_aleft, * d_atop, * d_aright, * d_abottom;
    float* d_bleft, * d_btop, * d_bright, * d_bbottom;
    float* d_iou;

    size_t size = num_boxes * sizeof(float);

    // Allocate device memory
    cudaMalloc(&d_aleft, size);
    cudaMalloc(&d_atop, size);
    cudaMalloc(&d_aright, size);
    cudaMalloc(&d_abottom, size);

    cudaMalloc(&d_bleft, size);
    cudaMalloc(&d_btop, size);
    cudaMalloc(&d_bright, size);
    cudaMalloc(&d_bbottom, size);

    cudaMalloc(&d_iou, size);

    // Copy data from host to device
    cudaMemcpy(d_aleft, h_aleft.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_atop, h_atop.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_aright, h_aright.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_abottom, h_abottom.data(), size, cudaMemcpyHostToDevice);

    cudaMemcpy(d_bleft, h_bleft.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_btop, h_btop.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bright, h_bright.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bbottom, h_bbottom.data(), size, cudaMemcpyHostToDevice);

    // Launch the kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_boxes + threadsPerBlock - 1) / threadsPerBlock;
    calculateIoU << <blocksPerGrid, threadsPerBlock >> > (
        d_aleft, d_atop, d_aright, d_abottom,
        d_bleft, d_btop, d_bright, d_bbottom,
        d_iou, num_boxes
        );

    // Copy result back to host
    cudaMemcpy(h_iou.data(), d_iou, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_aleft);
    cudaFree(d_atop);
    cudaFree(d_aright);
    cudaFree(d_abottom);

    cudaFree(d_bleft);
    cudaFree(d_btop);
    cudaFree(d_bright);
    cudaFree(d_bbottom);

    cudaFree(d_iou);

    // Print IoU results
    for (int i = 0; i < num_boxes; ++i) {
        std::cout << "IoU[" << i << "] = " << h_iou[i] << std::endl;
    }

    return 0;
}
