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

namespace cud {
    __global__ void compare_bool(bool* dst, const unsigned char* src, const int compare_num) {
        int blockIndex = threadIdx.x;
        int globalIndex = threadIdx.x + blockDim.x * blockIdx.x;

        *(dst + globalIndex) = *(src + globalIndex) > compare_num;
#if DEBUG
        if (blockIndex == 0) {
            printf("block idx : %d, src : %d,result : %d \n", blockIndex, *(src + globalIndex), *(dst + globalIndex));
        }
#endif // DEBUG


    }
}

cudaError_t img_automation_cuda(double** dst, unsigned char** src);
cudaError_t img_num_compare_call(bool* dst, unsigned char* src, int size, int compare_num);



