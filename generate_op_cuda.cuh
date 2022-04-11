#pragma warning(disable : 4996)

#include <iostream>
#include <filesystem>
#include <ctime>
#include <direct.h>		//mkdir
#include <errno.h>		//errno
#include <stdio.h>
#include <direct.h>	//getcwd
#include <omp.h>
#include <stdlib.h>
#include "opencv2/opencv.hpp"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "pcl/io/pcd_io.h"
#include "pcl/pcl_macros.h"
#include "pcl/point_types.h"

#define DEBUG 0
#define MAX_BLOCK_NUM 512
#define NORM_BLOCK_NUM 256
#define BLOCK_SIZE 16

// image size.
#define HEIGHT 480
#define WIDTH 640
#define CHANNEL 3

using namespace std;
using namespace cv;
using namespace pcl;

#define PCL_ADD_UNION_POINT4D_DOUBLE \
    union EIGEN_ALIGN16 { \
      double data[4]; \
      struct { \
        double x; \
        double y; \
        double z; \
      }; \
    };

struct EIGEN_ALIGN16 PointXYZRGB_double
{
    PCL_ADD_UNION_POINT4D_DOUBLE;
    PCL_ADD_RGB;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZRGB_double,
    (double, x, x)
    (double, y, y)
    (double, z, z)
    (std::uint32_t, rgba, rgba)
)

__global__ void img_op(double* z, unsigned char* depth_img, unsigned char* mask_img, double far_, double near_, float mask_threshold);
__global__ void point_op(double* dst_points, unsigned char* dst_point_colors, unsigned char* src_rgb, double* src_z, double* src_inverse_k);


cudaError_t img_op_kernel_call(double* dst_z, unsigned char* src_depth_img, unsigned char* scr_mask_img);
cudaError_t point_op_kernel_call(double** dst_points, unsigned char** dst_point_colors, unsigned char* src_rgb, double* src_z);

void trans_automation_cuda(double** dst_point, unsigned char** dst_point_color, unsigned char** src_images);
void depth_to_pcd(int img_set_size, string dataset_path);

