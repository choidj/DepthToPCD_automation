#include "generate_op_cuda.cuh"


int main() {
	char path_buf[_MAX_PATH] = { 0, };
	char dataset_path_[_MAX_PATH] = { 0, };
	char pcd_path_[_MAX_PATH] = { 0, };
	char* path_buf_ptr = getcwd(path_buf, _MAX_PATH);

	string temp = path_buf;

	string dataset_path = temp + "\\dataset\\";
	string pcd_path = dataset_path + "pcd_result";

	strcpy(dataset_path_, dataset_path.c_str());
	strcpy(pcd_path_, pcd_path.c_str());

	mkdir(pcd_path_);


	printf("pcd_path, dataset_path : %s, %s\n", pcd_path.c_str(), dataset_path.c_str());
	depth_to_pcd(1, dataset_path, pcd_path);

	return 0;
}


// dataset format : depth_0.png, depth_1.png ,...,
// result format : pc_0.png, pc_1.png ,...,
void depth_to_pcd(int img_set_size, string str_path, string dst_path) {
	string objs_names[] = { "depth_", "rgb_", "mask_", "pc_"};		// image materials for making PCD.
	string temp_name;
	string result_path;
	char num_buf[56];

	int cur_idx = 0;

	int pixel_size = HEIGHT * WIDTH * CHANNEL;
	int images_size = 3;

	clock_t start, end;

	Mat* src_imgs = new Mat[3];
	PointCloud <PointXYZRGB_double> point_cloud;

	//pcl::visualization::CloudViewer viewer("Check Cloud Viewer");

	//---------------------------------------------------------------------------------------
	// img mat allocation. 2-Dim. * is images (e.g., depth, rgb, mask), ** is image's pixels.
	unsigned char** images;
	images = (unsigned char**)malloc(images_size * sizeof(unsigned char*));
	*(images) = (unsigned char*)malloc(images_size * pixel_size * sizeof(unsigned char));
	// check if malloc is failed...
	if (images == NULL) { printf("images is falied to allocation.\n"); return ; }
	for (int i = 1; i < images_size; i++) {
		*(images + i) = *(images + i - 1) + pixel_size;
	}
	//----------------------------------------------------------------------------------------
	// dst_points, dst_point_color allocation...
	double** dst_points;
	unsigned char** dst_points_color;
	dst_points = (double**)malloc(HEIGHT * WIDTH * sizeof(double*));
	*(dst_points) = (double*)malloc(CHANNEL * HEIGHT * WIDTH * sizeof(double));

	dst_points_color = (unsigned char**)malloc(HEIGHT * WIDTH * sizeof(unsigned char*));
	*(dst_points_color) = (unsigned char*)malloc(CHANNEL * HEIGHT * WIDTH * sizeof(unsigned char));

	if (dst_points == NULL || dst_points_color == NULL) { printf("dst is falied to allocation.\n"); return ; }
	for (int i = 1; i < HEIGHT * WIDTH; i++) {
		*(dst_points_color + i) = *(dst_points_color + i - 1) + CHANNEL;
		*(dst_points + i) = *(dst_points + i - 1) + CHANNEL;
	}
	//----------------------------------------------------------------------------------------
	
	//pcd generation start.....
	for (int cur_data = 0; cur_data < img_set_size; cur_data++) {
		// allocate image Mats and read opencv imread..-------------------------------------------
		for (int i = 0; i < images_size; i++) {
			string temp_name;
			sprintf_s(num_buf, "%d", cur_data);
			temp_name = *(objs_names + i) + num_buf + ".png";
			*(src_imgs + i) = imread(str_path + temp_name);
			memset(num_buf, 0, 56);
		}

		// opencv Mat convert into usigned char..-------------------------------------------------
		for (int i = 0; i < images_size; i++) {
			memcpy(*(images + i), (src_imgs + i)->data, pixel_size * sizeof(unsigned char));
		}
		// check converting complete?
#if DEBUG
		for (int j = 0; j < 3; j++) {
			if (j == 0)
				printf("Depth image check....\n");
			if (j == 1)
				printf("RGB image check....\n");
			if (j == 2)
				printf("Mask image check....\n");
			for (int i = 0; i < 6; i++)
				printf("image pixel value : %d\n", *(*(images + j) + i));
		}
#endif	
		printf("Start converting 3 images (depth, rgb, mask) to point cloud....\n");
		start = clock();
		// automation start...--------------------------------------------------------------------
		trans_automation_cuda(dst_points, dst_points_color, images);

		cur_idx = 0;
		for (int i = 0; i < HEIGHT * WIDTH; i++) {
			if (*(*(dst_points + i)) == NULL)
				continue;
			// x y z setting,,,
			PointXYZRGB_double point;
			point.x = *(*(dst_points + i));
			point.y = *(*(dst_points + i) + 1);
			point.z = *(*(dst_points + i) + 2);

			// r g b setting,,,
			std::uint8_t r(*(*(dst_points_color + i)))\
				, g(*(*(dst_points_color + i) + 1))\
				, b(*(*(dst_points_color + i) + 2));

			std::uint32_t rgb = (static_cast<std::uint32_t>(r) << 16 |
				static_cast<std::uint32_t>(g) << 8 | static_cast<std::uint32_t>(b));
			point.rgb = *reinterpret_cast<float*>(&rgb);

			point_cloud.push_back(point);

#if DEBUG

			if (cur_idx == 0) {
				printf("x, y, z : %lf ,%lf, %lf || r, g, b : %d, %d, %d\n", \
					point.x, point.y, point.z, point.r, point.g, point.b);
				cur_idx++;
			}
#endif
		}
		
		sprintf_s(num_buf, "%d", cur_data);
		temp_name = *(objs_names + 3) + num_buf + ".png";
		memset(num_buf, 0, 56);
		result_path = dst_path + "\\" + "pc_0.pcd";
		io::savePCDFileASCII<PointXYZRGB_double>(result_path, point_cloud);

		end = clock();

		printf("Completed converting 3 images (depth, rgb, mask) to point cloud!!!\n");
		printf("Result PCD File Path : %s\n", result_path.c_str());  //will use os path 
		printf("elapsed time : %.2lf s\n", difftime(end, start) / 1000.0);
	}
	

	// img mat deallocation.
	free(*images); 				free(images);
	free(*dst_points);			free(dst_points);
	free(*dst_points_color);	free(dst_points_color);

}