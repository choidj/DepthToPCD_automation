#include "transform_op_cuda.cuh"


// Helper function for using CUDA to add vectors in parallel.
cudaError_t img_op_kernel_call(double* dst_z, unsigned char* src_depth_img, unsigned char* src_mask_img) {
    double* dev_z; unsigned char* dev_depth_img; unsigned char* dev_mask_img;

    double far_ = 5; double near_ = 0.3; float mask_threshold = 10;

    int size = HEIGHT * WIDTH;

    cudaError_t cudaStatus;

    int img_grid1D = ceil((float)size / (float)MAX_BLOCK_NUM);

    dim3 grid(img_grid1D, 1, 1); dim3 block(MAX_BLOCK_NUM, 1, 1);

#if DEBUG
    for (int i = 0; i < 30; i++) {
        printf("[i : %d] depth src : %d, mask src : %d\n", i, *(src_depth_img +i), *(src_mask_img + i));
    }
#endif

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_z, size * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed on dev_z!");
        goto Error;
    }
    
    cudaStatus = cudaMalloc((void**)&dev_depth_img, CHANNEL * size * sizeof(unsigned char));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed on dev_depth_img!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_mask_img, CHANNEL * size * sizeof(unsigned char));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed on dev_mask_img!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_depth_img, src_depth_img, CHANNEL * size * sizeof(unsigned char), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed on dev_depth_img!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_mask_img, src_mask_img, CHANNEL * size * sizeof(unsigned char), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed dev_mask_img!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    img_op << <grid, block >> > (dev_z, dev_depth_img, dev_mask_img, far_, near_, mask_threshold);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(dst_z, dev_z, size * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed on z!");
        goto Error;
    }

Error:
    cudaFree(dev_z); cudaFree(dev_depth_img); cudaFree(dev_mask_img);

    return cudaStatus;
}

cudaError_t point_op_kernel_call(double** dst_points, unsigned char** dst_point_colors, unsigned char* src_rgb, double* src_z) {
    // need to add inverse op???
    double inverse_k[][3] = {
        {-0.00174699220352319, 0 ,0.559037505127422},
        {0, -0.00174346879155994, 0.418432509974385},
        {0, 0, 1}
    };
    int k_size = 9;

    double* dev_inverse_k; double* dev_z; double* dev_points; unsigned char* dev_point_colors; unsigned char* dev_rgb;

    int size = HEIGHT * WIDTH;

    cudaError_t cudaStatus;

    int grid2D_x = ceil((float)WIDTH / (float)NORM_BLOCK_NUM);
    int grid2D_y = HEIGHT;

    dim3 grid(grid2D_x, grid2D_y, 1); dim3 block(NORM_BLOCK_NUM, 1, 1);

#if DEBUG
    int inx = 0;
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            if (*(src_z + i * WIDTH + j) != 1.3) {
                printf("height : %d, width : %d, rgb src : %d, z src : %lf\n", i, j, *(src_rgb + i), *(src_z + i * WIDTH + j));
                if (*(src_z + i * WIDTH + j) < 0)
                    inx++;
            }
        }
    }
    printf("total - point : %d\n", inx);
#endif

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_z, size * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed on dev_z!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_rgb, CHANNEL * size * sizeof(unsigned char));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed on dev_rgb!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_points, CHANNEL * size * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed on dev_points!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_point_colors, CHANNEL * size * sizeof(unsigned char));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed on dev_point_colors!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_inverse_k, k_size * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed on dev_inverse_k!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_inverse_k, inverse_k, k_size * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed on dev_inverse_k!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_z, src_z, size * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed dev_z!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_rgb, src_rgb, CHANNEL * size * sizeof(unsigned char), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed dev_rgb!");
        goto Error;
    }
    
    // Launch a kernel on the GPU with one thread for each element.
    point_op << <grid, block >> > (dev_points, dev_point_colors, dev_rgb, dev_z, dev_inverse_k);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(*dst_points, dev_points, CHANNEL * size * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed on z!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(*dst_point_colors, dev_point_colors, CHANNEL * size * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed on z!");
        goto Error;
    }

Error:
    cudaFree(dev_z); cudaFree(dev_inverse_k); 
    cudaFree(dev_points); cudaFree(dev_point_colors);
    cudaFree(dev_rgb);

    return cudaStatus;
}


// depth image : *(src), rgb image : *(src + 1), mask image : *(src + 2)
void trans_automation_cuda(double** dst_point, unsigned char** dst_point_color, unsigned char** src_images) {
    cudaError_t cudaStatus;
    int pixel_size = HEIGHT * WIDTH * CHANNEL;
    double* z = (double*)malloc(HEIGHT * WIDTH * sizeof(double));

#if DEBUG
    for (int i = 0; i < 30; i++) {
        printf("[i : %d] depth src : %d, rgb src : %d, mask src : %d\n", i, *(*(src_images) + i), *(*(src_images + 1) + i), *(*(src_images + 2) + i));
    }
#endif

    // image operation kernel call  matlab :-- pts = zeros(height*width, 3) color = uint8(zeros(height * width, 3))--
    cudaStatus = img_op_kernel_call(z, *(src_images), *(src_images + 2));

    //point operation kernel call here..
    cudaStatus = point_op_kernel_call(dst_point, dst_point_color, *(src_images + 1), z);

    free(z);
}
