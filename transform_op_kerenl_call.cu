#include "transform_op_cuda.cuh"

// Helper function for using CUDA to add vectors in parallel.
cudaError_t img_op_kernel_call(double* z, unsigned char* depth_img, unsigned char* mask_img) {
    double* dev_z; unsigned char* dev_depth_img; unsigned char* dev_mask_img;

    double far = 5; double near = 0.3; float mask_threshold = 10;

    int size = HEIGHT * WIDTH;
    cudaError_t cudaStatus;

    int img_grid1D = ceil((float)size / (float)MAX_BLOCK_NUM);

    dim3 grid(img_grid1D, 1, 1); dim3 block(MAX_BLOCK_NUM, 1, 1);

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
    
    cudaStatus = cudaMalloc((void**)&dev_depth_img, size * sizeof(unsigned char));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed on dev_depth_img!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_mask_img, size * sizeof(unsigned char));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed on dev_mask_img!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_depth_img, depth_img, size * sizeof(unsigned char), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed on dev_depth_img!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_depth_img, depth_img, size * sizeof(unsigned char), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed dev_mask_img!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    img_op << <grid, block >> > (dev_z, dev_depth_img, dev_mask_img, far, near, mask_threshold);

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
    cudaStatus = cudaMemcpy(z, dev_z, size * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed on z!");
        goto Error;
    }

Error:
    cudaFree(dev_z); cudaFree(dev_depth_img); cudaFree(dev_mask_img);

    return cudaStatus;
}


// depth image : *(src), rgb image : *(src + 1), mask image : *(src + 2)
cudaError_t trans_automation_cuda(double** dst, unsigned char** src) {
    cudaError_t cudaStatus;
    int pixel_size = HEIGHT * WIDTH * CHANNEL;

    double inverse_k[][3] = { 
        {-0.00174699220352319, 0 ,0.559037505127422},
        {0, -0.00174346879155994, 0.418432509974385},
        {0, 0, 1}
    };


    double* z = (double*)malloc(HEIGHT * WIDTH * sizeof(double));



    // image operation kernel call  matlab :-- pts = zeros(height*width, 3) color = uint8(zeros(height * width, 3))--
    cudaStatus = img_op_kernel_call(z, *(src + 1), *(src + 2));

    //point operation kernel call



    std::free(z);
    for (int i = 0; i < 3; i++)
        delete(inverse_k[i]);
    delete(inverse_k);

    return cudaStatus;
}
