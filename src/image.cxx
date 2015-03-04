#include "image.h"

void Image::saveImage() {
	cv::imwrite(filename.c_str(), image, compression_params);
}