#ifndef TPS_IMAGE_H_
#define TPS_IMAGE_H_

#include <vector>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace tps {

class Image {
public:
	Image(std::string filename) {
		filename_ = filename;
		image = cv::imread(filename_, CV_LOAD_IMAGE_GRAYSCALE);
	  if( !image.data )
  	{ std::cout << " --(!) Error reading images form file" << filename << std::endl; }
		dimensions.push_back(image.rows);
		dimensions.push_back(image.cols);
		compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
		compression_params.push_back(95);
	}
	Image(int rows, int cols, const std::string& filename) {
		filename_ = filename;
		image = cv::Mat::zeros(rows, cols, CV_8U);
		dimensions.push_back(image.rows);
		dimensions.push_back(image.cols);
		compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
		compression_params.push_back(95);
	}
	void save() {cv::imwrite(filename_.c_str(), image, compression_params);};
	cv::Mat getImage() {return image;};
	std::vector<int> getDimensions() { return dimensions; };
	template<typename T> T* getRowPtr(int row) { return image.ptr<T>(row); };
	template<typename T> void changePixelAt(int row, int col, T value);
	template<typename T> T getPixelAt(int row, int col);
	template<typename T> T bilinearInterpolation(float row, float col);
	template<typename T> T NNInterpolation(float row, float col);
private:
	friend class Surf;
	cv::Mat image;
	std::vector<int> dimensions;
	std::string filename_;
	std::vector<int> compression_params;
	int getNearestInteger(float number) {
	if ((number - floor(number)) <= 0.5) return floor(number);
	return floor(number) + 1.0;
}
};

template<typename T> void Image::changePixelAt(int row, int col, T value) {
	if (row >= 0 && row < dimensions[0]-1 && col >= 0 && col < dimensions[1]-1)
		image.at<T>(row, col) = value;
}

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

template<typename T>
T Image::bilinearInterpolation(float row, float col) {
	int u = trunc(row);
	int v = trunc(col);
	uchar pixelOne = getPixelAt<uchar>(u, v);
	uchar pixelTwo = getPixelAt<uchar>(u+1, v);
	uchar pixelThree = getPixelAt<uchar>(u, v+1);
	uchar pixelFour = getPixelAt<uchar>(u+1, v+1);

	T interpolation = (u+1-row)*(v+1-col)*pixelOne
												+ (row-u)*(v+1-col)*pixelTwo 
												+ (u+1-row)*(col-v)*pixelThree
												+ (row-u)*(col-v)*pixelFour;
	return interpolation;
}

template<typename T>
T Image::NNInterpolation(float row, float col) {
	int nearRow = getNearestInteger(row);
	int nearCol = getNearestInteger(col);
	T aux = getPixelAt<uchar>(nearRow, nearCol);
	return aux;
}

} // namespace

#endif