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

__global__ void point_op(double* dst_points, unsigned char* dst_point_colors, unsigned char* src_rgb, double* src_z, double* src_inverse_k) {
    int blockIndex = threadIdx.x + blockIdx.x * blockDim.x;

    if (blockIndex < WIDTH) {
        int globalIndex = blockIndex + blockIdx.y * WIDTH;
        int u = blockIndex;
        int v = blockIdx.y;

        *(dst_points + (globalIndex * 3)) = *(src_z + globalIndex) * *(src_inverse_k) * u +  \
            *(src_z + globalIndex) * *(src_inverse_k + 1) * v + \
            *(src_z + globalIndex) * *(src_inverse_k + 2);
        *(dst_points + (globalIndex * 3) + 1) = *(src_z + globalIndex) * *(src_inverse_k + 4)*u + \
            * (src_z + globalIndex) * *(src_inverse_k + 5) * v + \
            * (src_z + globalIndex) * *(src_inverse_k + 6);
        *(dst_points + (globalIndex * 3) + 2) = *(src_z + globalIndex);

        *(dst_point_colors + (globalIndex * 3)) = *(src_rgb + globalIndex);
        *(dst_point_colors + (globalIndex * 3) + 1) = *(src_rgb + globalIndex + 1);
        *(dst_point_colors + (globalIndex * 3) + 2) = *(src_rgb + globalIndex + 2);


#if DEBUG
        if (globalIndex == 250000) {
            printf("global idx : %d, z : %lf, mask_img : %d , depth_img : %d\n", globalIndex, *(z + globalIndex), *(mask_img + globalIndex), *(depth_img + globalIndex));
        }

#endif // DEBUG
    }
}
