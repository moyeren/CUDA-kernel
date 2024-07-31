#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

__global__ void hwc2chw(float* src, float* dst,
	int batch_size, int img_height, int img_width, int img_area, int img_volume)
{
	int dx = blockDim.x * blockIdx.x + threadIdx.x;
	int dy = blockDim.y * blockIdx.y + threadIdx.y;
	if (dx < img_volume && dy < batch_size)
	{
		int ch = dx / img_area; // 0 1 2
		assert(ch < 3);
		int sub_idx = dx % img_area;
		int row = sub_idx / img_width;
		int col = sub_idx % img_width;

		int dx_ = row * (img_width * 3) + col * 3 + ch;
		dst[dy * img_volume + dx] = src[dy * img_volume + dx_];
		//printf("[x = %d; y = %d] \n", dx, dy);
	}
}
