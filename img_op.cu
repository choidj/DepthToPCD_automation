#include "img_op.cuh"





// Helper function for using CUDA to add vectors in parallel.
cudaError_t img_num_compare_call(bool* dst, unsigned char* src, int size, int compare_num) {
    unsigned char* dev_src = 0;
    bool* dev_dst = 0;
    
    cudaError_t cudaStatus;

    int img_grid1D = ceil((float)size / (float)MAX_BLOCK_NUM);

    dim3 grid(img_grid1D, 1, 1);
    dim3 block(MAX_BLOCK_NUM, 1, 1);

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_dst, size * sizeof(bool));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    
    cudaStatus = cudaMalloc((void**)&dev_src, size * sizeof(unsigned char));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_src, src, size * sizeof(unsigned char), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    cud::compare_bool<< <grid, block >> > (dev_dst, dev_src, compare_num);

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
    cudaStatus = cudaMemcpy(dst, dev_dst, size * sizeof(bool), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_src);
    cudaFree(dev_dst);

    return cudaStatus;
}

cudaError_t img_automation_cuda(double** dst, unsigned char** src) {
    int pixel_size = HEIGHT * WIDTH * CHANNEL;

    unsigned char* src_buffer = (unsigned char*)malloc(pixel_size * sizeof(unsigned char));
    bool* dst_buffer = (bool*)malloc(pixel_size * sizeof(bool));



    free(src_buffer);
    free(dst_buffer);
}
