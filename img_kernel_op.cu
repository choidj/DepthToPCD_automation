#include "img_op_cuda.cuh"


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