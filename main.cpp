#include "transform_op_cuda.cuh"


int main() {
	string img_names[] = {"depth_", "rgb_", "mask_"};		// image materials for making PCD.
	string path = ".\\DepthToPCD\\";									// relative path.
	char num_buf[256];
	int pixel_size = HEIGHT * WIDTH * CHANNEL;
	size_t material_size = sizeof(img_names) / sizeof(string);
	Mat* src_imgs = new Mat[material_size];

	//---------------------------------------------------------------------------------------
	// img mat allocation. 2-Dim. * is images (e.g., depth, rgb, mask), ** is image's pixels.
	unsigned char** images;
	images = (unsigned char**)malloc(material_size * sizeof(unsigned char*));
	*(images) = (unsigned char*)malloc(material_size * pixel_size * sizeof(unsigned char));
	// check if malloc is failed...
	if (images == NULL) { printf("images is falied to allocation.\n"); return -1; }
	for (int i = 1; i < material_size; i++) {
		*(images + i) = *(images + i - 1) + pixel_size;
	}
	//----------------------------------------------------------------------------------------
	// dst_points, dst_point_color allocation...
	double** dst_points;
	unsigned char** dst_points_color;
	dst_points = (double**)malloc(3 * sizeof(double*));
	*(dst_points) = (double*)malloc(pixel_size * sizeof(double));

	dst_points_color = (unsigned char**)malloc(3 * sizeof(unsigned char*));
	*(dst_points_color) = (unsigned char*)malloc(pixel_size * sizeof(unsigned char));

	if (dst_points == NULL || dst_points_color == NULL) { printf("dst is falied to allocation.\n"); return -1; }
	for (int i = 1; i < 3; i++) {
		*(dst_points_color + i) = *(dst_points_color + i - 1) + HEIGHT * WIDTH;
		*(dst_points + i) = *(dst_points + i - 1) + HEIGHT * WIDTH;
	}
	//----------------------------------------------------------------------------------------
	int cur_idx = 0;

	// allocate image Mats and read opencv imread..-------------------------------------------
	for (int i = 0; i < material_size; i++) {
		string temp_name;
		sprintf_s(num_buf, "%d", cur_idx);
		temp_name = *(img_names + i) + num_buf + ".png";
		cout << "path : " << path + temp_name << endl;
		*(src_imgs + i) = imread(path + temp_name);
	}

	// opencv Mat convert into usigned char..-------------------------------------------------
	for (int i = 0; i < material_size; i++) {
		memcpy(*(images + i), (src_imgs + i)->data, pixel_size * sizeof(unsigned char));
	}

	// automation start...--------------------------------------------------------------------
	trans_automation_cuda(dst_points, dst_points_color, images);

	imshow("befo", *(src_imgs + 2));
	//imshow("Depth", src_imgs[0]);
	//imshow("RGB", src_imgs[1]);
	//imshow("Mask", src_imgs[2]);

	waitKey(50000);


	// img mat deallocation.
	free(*images); 				free(images);
	free(*dst_points);			free(dst_points);
	free(*dst_points_color);	free(dst_points_color);
	delete(src_imgs);

	return 0;
}