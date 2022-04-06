#include "img_op_cuda.cuh"


int main() {
	string img_names[] = {"depth_", "rgb_", "mask_"};		// image materials for making PCD.
	string path = ".\\DepthToPCD\\";									// relative path.
	char num_buf[256];
	int pixel_size = HEIGHT * WIDTH * CHANNEL;
	size_t material_size = sizeof(img_names) / sizeof(string);
	Mat* src_imgs = new Mat[material_size];

	unsigned char** src_buffer;
	src_buffer = (unsigned char**)malloc(material_size * sizeof(unsigned char*));
	*(src_buffer) = (unsigned char*)malloc(material_size * pixel_size * sizeof(unsigned char));
	for (int i = 1; i < material_size; i++) {
		*(src_buffer + i) = *(src_buffer + i - 1) + pixel_size;
	}

	double** dst_buffer = 0;
	int cur_idx = 0;

	// allocate image Mats.
	for (int i = 0; i < material_size; i++) {
		string temp_name;
		sprintf_s(num_buf, "%d", cur_idx);
		temp_name = img_names[i] + num_buf + ".png";
		cout << "path : " << path + temp_name << endl;
		src_imgs[i] = imread(path + temp_name);
	}


	for (int i = 0; i < material_size; i++) {
		memcpy(*(src_buffer + i), (src_imgs + i)->data, pixel_size * sizeof(unsigned char));
	}
	
	cudaError_t cudaStatus = img_automation_cuda(dst_buffer, src_buffer);

	//imshow("befo", src_imgs[2]);
	//imshow("after", dst_result);
	//imshow("Depth", src_imgs[0]);
	//imshow("RGB", src_imgs[1]);
	//imshow("Mask", src_imgs[2]);

	waitKey(50000);

	free(*src_buffer);
	free(src_buffer);

	return 0;
}