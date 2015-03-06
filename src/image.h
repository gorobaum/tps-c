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

	Image(int rows, int cols, std::string filename) {
		filename_ = filename;
		image = cv::Mat::zeros(rows, cols, CV_8U);
		dimensions.push_back(image.rows);
		dimensions.push_back(image.cols);
		compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
		compression_params.push_back(95);
	}
	void save() {cv::imwrite(filename_.c_str(), image, compression_params);};
	template<typename T> T getPixelAt(int row, int col);
	cv::Mat getImage() {return image;};
	std::vector<int> getDimensions() { return dimensions; };
private:
	cv::Mat image;
	std::vector<int> dimensions;
	std::string filename_;
	std::vector<int> compression_params;
};

template<typename T>
T Image::getPixelAt(int row, int col) {

	if (row > dimensions[0]-1 || row < 0)
		return 0;
	else if (col > dimensions[1]-1 || col < 0)
		return 0;
	else {
		return image.at<T>(row, col);
	}
}

#endif