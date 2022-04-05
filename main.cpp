#include "img_op.cuh"


int main() {
	int height = 480; int width = 640; int pixel_size = height * width * 3;
	string img_names[] = {"depth_", "rgb_", "mask_"};		// image materials for making PCD.
	string path = ".\\DepthToPCD\\";									// relative path.
	char num_buf[256];

	size_t material_size = sizeof(img_names) / sizeof(string);
	Mat* src_imgs = new Mat[material_size];

	unsigned char* src_buffer = (unsigned char*)malloc(pixel_size * sizeof(unsigned char));
	bool* dst_buffer = (bool*)malloc(pixel_size * sizeof(bool));
	int cur_idx = 0;

	// allocate image Mats.
	for (int i = 0; i < material_size; i++) {
		string temp_name;
		sprintf_s(num_buf, "%d", cur_idx);
		temp_name = img_names[i] + num_buf + ".png";
		cout << "path : " << path + temp_name << endl;
		src_imgs[i] = imread(path + temp_name);
	}

	if (src_buffer == NULL) {
		printf("NULL Pointer ! \n");
	}

	memcpy(src_buffer, src_imgs[2].data, pixel_size * sizeof(unsigned char));
	
	cudaError_t cudaStatus = automation_cuda(dst_buffer, src_buffer);

	Mat dst_result(height, width, CV_8UC3, dst_buffer);
	//imshow("befo", src_imgs[2]);
	//imshow("after", dst_result);
	//imshow("Depth", src_imgs[0]);
	//imshow("RGB", src_imgs[1]);
	//imshow("Mask", src_imgs[2]);

	waitKey(50000);

	free(src_buffer);
	free(dst_buffer);

	return 0;
}