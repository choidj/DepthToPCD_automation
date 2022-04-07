#include "transform_op_cuda.cuh"


__global__ void img_op(double* z, unsigned char* depth_img, unsigned char* mask_img, double far, double near, float mask_threshold) {
    int globalIndex = threadIdx.x + blockDim.x * blockIdx.x;


    *(mask_img + globalIndex) = *(mask_img + globalIndex) > mask_threshold;

    *(z + globalIndex) = 1. - (far - near)*(((double)(*(depth_img + globalIndex)) / 255.) * (*(mask_img + globalIndex))) + near;

#if DEBUG
    if (globalIndex == 250000) {
        printf("global idx : %d, z : %lf, mask_img : %d , depth_img : %d\n", globalIndex, *(z + globalIndex), *(mask_img + globalIndex), *(depth_img + globalIndex));
    }
#endif // DEBUG
}