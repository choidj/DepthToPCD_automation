#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "opencv2/opencv.hpp"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define DEBUG 0
#define MAX_BLOCK_NUM 512
#define BLOCK_SIZE 16

// image size.
#define HEIGHT 640
#define WIDTH 480
#define CHANNEL 3

using namespace std;
using namespace cv;

__global__ void img_op(double* z, unsigned char* depth_img, unsigned char* mask_img, double far, double near, float mask_threshold);

cudaError_t trans_automation_cuda(double** dst, unsigned char** src);
cudaError_t img_op_kernel_call(double* z, unsigned char* depth_img, unsigned char* mask_img);






