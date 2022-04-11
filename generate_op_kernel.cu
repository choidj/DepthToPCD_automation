#include "generate_op_cuda.cuh"


__global__ void img_op(double* z, unsigned char* data, double far_, double near_, float mask_threshold) {
    int global_index = threadIdx.x + blockDim.x * blockIdx.x;
    int total_index = 3 * global_index + 2;

    if (global_index < (HEIGHT* WIDTH)) {
        int z_jump = blockIdx.z * HEIGHT * WIDTH;
        int data_jump = HEIGHT * WIDTH * CHANNEL * blockIdx.z;

        unsigned char* mask_img = data + data_jump * 2;
        unsigned char* depth_img = data + data_jump;

        *(mask_img + total_index) = *(mask_img + total_index) > mask_threshold;

        *(z + global_index + z_jump) = 1. - (far_ - near_) * (((double)(*(depth_img + total_index)) / 255.) * (*(mask_img + total_index))) + near_;
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
        int data_jump = HEIGHT * WIDTH * CHANNEL * blockIdx.z;
        int z_jump = blockIdx.z * HEIGHT * WIDTH;
        double u = blockIndex;
        double v = blockIdx.y;

        if (*(src_z + global_index + z_jump) < 0) {
            *(dst_points + total_index + data_jump) = *(src_z + global_index + z_jump) * *(src_inverse_k) * u + \
                *(src_z + global_index + z_jump) * *(src_inverse_k + 1) * v + \
                *(src_z + global_index + z_jump) * *(src_inverse_k + 2);
            *(dst_points + total_index + 1 + data_jump) = *(src_z + global_index + z_jump) * *(src_inverse_k + 3) * u + \
                *(src_z + global_index + z_jump) * *(src_inverse_k + 4) * v + \
                *(src_z + global_index + z_jump) * *(src_inverse_k + 5);
            *(dst_points + total_index + 2 + data_jump) = *(src_z + global_index + z_jump);

            *(dst_point_colors + total_index + data_jump) = *(src_rgb + total_index + 2 + data_jump);
            *(dst_point_colors + total_index + 1 + data_jump) = *(src_rgb + total_index + 1 + data_jump);
            *(dst_point_colors + total_index + 2 + data_jump) = *(src_rgb + total_index + data_jump);
        }
        else {
            *(dst_points + total_index + data_jump) = NULL;
            *(dst_points + total_index + 1 + data_jump) = NULL;
            *(dst_points + total_index + 2 + data_jump) = NULL;

            *(dst_point_colors + total_index + data_jump) = NULL;
            *(dst_point_colors + total_index + 1 + data_jump) = NULL;
            *(dst_point_colors + total_index + 2 + data_jump) = NULL;


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
