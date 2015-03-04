#ifndef TPS_IMAGE_H_
#define TPS_IMAGE_H_

#include <vector>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

class Image {
public:
	Image(std::string filename) {
		filename_ = filename;
		image = cv::imread(filename_, CV_LOAD_IMAGE_GRAYSCALE);
		dimensions.push_back(image.rows);
		dimensions.push_back(image.cols);
		compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
		compression_params.push_back(95);
	}
	void saveImage() {cv::imwrite(filename_.c_str(), image, compression_params);};
	template<typename T> T getPixelAt(std::vector<int> position);
	std::vector<int> getDimensions() { return dimensions; };
private:
	cv::Mat image;
	std::vector<int> dimensions;
	std::string filename_;
	std::vector<int> compression_params;
};

template<typename T>
T Image::getPixelAt(std::vector<int> position) {

	if (position[0] > dimensions[0]-1 || position[0] < 0)
		return 0;
	else if (position[1] > dimensions[1]-1 || position[1] < 0)
		return 0;
	else {
		return image.at<T>(position[0], position[1]);
	}
}

#endif