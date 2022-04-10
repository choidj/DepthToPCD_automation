#include "transform_op_cuda.cuh"


__global__ void img_op(double* z, unsigned char* depth_img, unsigned char* mask_img, double far_, double near_, float mask_threshold) {
    int global_index = threadIdx.x + blockDim.x * blockIdx.x;
    int total_index = 3 * global_index + 2;

    if (global_index < (HEIGHT* WIDTH)) {
        *(mask_img + total_index) = *(mask_img + total_index) > mask_threshold;

        *(z + global_index) = 1. - (far_ - near_) * (((double)(*(depth_img + total_index)) / 255.) * (*(mask_img + total_index))) + near_;
#if DEBUG
        if (global_index == 250000) {
            printf("global idx : %d, z : %lf, mask_img : %d , depth_img : %d\n", \
                global_index, *(z + global_index), *(mask_img + total_index), *(depth_img + total_index));
        }
#endif // DEBUG
    }
}

__global__ void point_op(double* dst_points, unsigned char* dst_point_colors, unsigned char* src_rgb, double* src_z, double* src_inverse_k) {
    int blockIndex = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (blockIndex < WIDTH) {
        int global_index = blockIndex + blockIdx.y * WIDTH;
        int total_index = 3 * global_index;

        double u = blockIndex;
        double v = blockIdx.y;

        if (*(src_z + global_index) < 0) {
            *(dst_points + total_index) = *(src_z + global_index) * *(src_inverse_k) * u + \
                *(src_z + global_index) * *(src_inverse_k + 1) * v + \
                *(src_z + global_index) * *(src_inverse_k + 2);
            *(dst_points + total_index + 1) = *(src_z + global_index) * *(src_inverse_k + 3) * u + \
                *(src_z + global_index) * *(src_inverse_k + 4) * v + \
                *(src_z + global_index) * *(src_inverse_k + 5);
            *(dst_points + total_index + 2) = *(src_z + global_index);

            *(dst_point_colors + total_index) = *(src_rgb + total_index + 2);
            *(dst_point_colors + total_index + 1) = *(src_rgb + total_index + 1);
            *(dst_point_colors + total_index + 2) = *(src_rgb + total_index);
        }
        else {
            *(dst_points + total_index) = NULL;
            *(dst_points + total_index + 1) = NULL;
            *(dst_points + total_index + 2) = NULL;

            *(dst_point_colors + total_index) = NULL;
            *(dst_point_colors + total_index + 1) = NULL;
            *(dst_point_colors + total_index + 2) = NULL;


        }
#if DEBUG
        if (blockIndex == 0) {
            printf("global idx : %d, z : %lf, point : %lf , rgb : %d\n", \
                global_index, *(src_z + global_index), *(dst_points + total_index),\
                *(dst_point_colors + total_index));

        }
#endif // DEBUG
    }
}
